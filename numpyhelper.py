import vapoursynth as vs
import numpy as np
from typing import Optional

core = vs.core


def vapoursynth_to_numpy(frame: vs.VideoFrame) -> np.ndarray:
    """Optimized conversion from Vapoursynth frame to numpy array."""
    return np.dstack([np.array(frame[i], copy=False) for i in range(frame.format.num_planes)])

def numpy_to_vapoursynth(array: np.ndarray, reference_frame: vs.VideoFrame) -> vs.VideoFrame:
    """Optimized conversion from numpy array to Vapoursynth frame."""
    if array.ndim == 2:
        array = array[..., np.newaxis]
    
    new_frame = reference_frame.copy()
    
    for i in range(new_frame.format.num_planes):
        plane = array[..., i]
        if not plane.flags['C_CONTIGUOUS']:
            plane = np.ascontiguousarray(plane)
        
        dest = np.array(new_frame[i], copy=False)
        np.copyto(dest, plane)
    
    return new_frame
    
def test_numpy_conversion(clip: vs.VideoNode, strict: bool = True) -> vs.VideoNode:
    """Test the numpy conversion roundtrip."""
    def _test(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        try:
            arr = vapoursynth_to_numpy(f)
            new_frame = numpy_to_vapoursynth(arr, f)
            
            for i in range(f.format.num_planes):
                if not np.array_equal(np.array(f[i]), np.array(new_frame[i])):
                    msg = f"Frame {n} plane {i} mismatch!"
                    if strict:
                        raise ValueError(msg)
                    print("Warning:", msg)
                    return f.text.Text("Conversion Error").get_frame(0)
            
            return new_frame
        except Exception as e:
            if strict:
                raise
            print(f"Error processing frame {n}: {str(e)}")
            return f.text.Text("Error").get_frame(0)
    
    return clip.std.ModifyFrame(clip, _test)
        
def AutoFixLineBorders(clip: vs.VideoNode, threshold: int = 24, bordersize: int = 4, debug: bool = False, maxShift: int = 60) -> vs.VideoNode:
    if clip.format.id != vs.YUV444P8:
        raise ValueError("AutoFixLineBorders: Only YUV444P8 input supported.")
      
    def process_frame(n, clip, threshold, bordersize, debug, maxShift):
        f = clip.get_frame(n)
        arr = vapoursynth_to_numpy(f)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]

        y = arr[..., 0]
        u = arr[..., 1]
        v = arr[..., 2]
        height, width = y.shape

        # Finde erstes nicht-dunkles Pixel pro Zeile
        first_non_dark = np.argmax(y > threshold, axis=1)
        first_non_dark[y.max(axis=1) <= threshold] = 0  # Sonderfall: ganze Zeile dunkel

        # Berechne Shift
        shifts = np.clip(first_non_dark - bordersize, 0, width)

        # maxShift-Logik: wenn eine Zeile zu viel verschieben müsste, einfach nicht verschieben
        shifts = np.where(first_non_dark > maxShift, 0, shifts)

        if debug:
            new_y = np.where(shifts[:, None] > 0, 128, y)
            new_u = np.where(shifts[:, None] > 0, 255, u)
            new_v = np.where(shifts[:, None] > 0, 128, v)
        else:
            new_y = np.full_like(y, 16)
            new_u = np.full_like(u, 128)
            new_v = np.full_like(v, 128)
            for row in range(height):
                shift = shifts[row]
                if shift > 0:
                    new_y[row, :-shift] = y[row, shift:]
                    new_u[row, :-shift] = u[row, shift:]
                    new_v[row, :-shift] = v[row, shift:]
                else:
                    # Keine Verschiebung: Original übernehmen
                    new_y[row, :] = y[row, :]
                    new_u[row, :] = u[row, :]
                    new_v[row, :] = v[row, :]

        result = np.dstack((new_y, new_u, new_v)).astype(np.uint8)
        return numpy_to_vapoursynth(result, f)
    
    return clip.std.ModifyFrame(
        clips=[clip],
        selector=lambda n, f: process_frame(n, clip, threshold, bordersize, debug, maxShift))
