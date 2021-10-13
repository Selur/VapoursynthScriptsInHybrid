import vapoursynth as vs
from vapoursynth import core
import functools
import havsfunc


class Crossfade:
    '''
    Crossfades from one function into other on a clip.
    This is not for crossfading clips.
    One Function passed as None causes to fade into a filter or fade out from a filter
    example:
    
    import vapoursynth as vs
    import animate
    clip = ....
    def headline1(clip, *args):       return clip.text.Text('Our Headline')
    def headline2(clip, *args):       return clip.text.Text('... and other headline')
    MAP = [
                 (0, 100),     [headline1],
                 (101,150),     [animate.Crossfade(headline1, headline2)],
                 (151,200),     [headline2],
          ]
    clip_out = animate.run(clip, MAP)   
    '''
    
    def __init__(self, function1=None, function2=None):           
        self.function1 = function1
        self.function2 = function2
                      
    def __call__(self, clip, n, lower, upper):
        return core.std.Merge(
                                self.function1(clip) if self.function1 is not None else clip,
                                self.function2(clip) if self.function2 is not None else clip,
                                (n-lower)/(upper-lower)
                             )

     
class CrossfadeFromColor:
    '''
    crossfades color to clip
    example:
    
    import vapoursynth as vs
    import animate
    clip = ....
    MAP = [
            (0,  60),  [animate.CrossfadeFromColor( (16,128,128) )],
          ]
    clip_out = animate.run(clip, MAP)    
    '''
    
    def __init__(self, color):
        self.color = color       
        
    def __call__(self, clip, n, lower, upper):
        return core.std.Merge(clip.std.BlankClip(color=self.color), clip, (n-lower)/(upper-lower))

        
class CrossfadeToColor:
    '''
    crossfades clip to color
    example:

    import vapoursynth as vs
    import animate
    clip = ....
    end = clip.num_frames-1
    MAP = [
            (end-60, end),  [animate.CrossfadeToColor( (16,128,128) )
          ]
    clip_out = animate.run(clip, MAP  )     
    '''
    
    def __init__(self, color):
        self.color = color       
        
    def __call__(self, clip, n, lower, upper):
        return core.std.Merge(clip, clip.std.BlankClip(color=self.color, length=1), (n-lower)/(upper-lower))

class Arguments:
    '''
    animates just selected arguments for a filter,
    example:
    
    import vapoursynth as vs
    import animate
    import adjust
    clip = ....
    TWEAK1 =      dict(hue=0.0,  sat=1.3,    bright=8.0,   cont=1.1,    coring=True)
    TWEAK2 =      dict(hue=0.0,  sat=1.4,    bright=15.0,  cont=1.3,    coring=True)
    #declare what arguments you want to animate, int of float, but None to not animate an argument:
    TWEAK_TYPES = dict(hue=None, sat=float,  bright=float, cont=float,  coring=None)
    def tweak1(clip,*args):
        return adjust.Tweak(clip, **TWEAK1)
    def tweak2(clip,*args):
        return adjust.Tweak(clip, **TWEAK2)
    MAP = [
             (0, 100),                 [tweak1],
             (101, 150),               [animate.Arguments(adjust.Tweak, TWEAK1, TWEAK2, TWEAK_TYPES)],
             (151, clip.num_frames-1), [tweak2]
          ]
    clip_out = animate.run(clip, MAP)       
    '''
    
    def __init__(self, function, set1, set2, types):       
        self.function = function
        self.set1   = set1
        self.types  = types
        self.diff   = {key : round(value-set2[key]) if types[key] == int else round(value-set2[key], 2) for key, value in set1.items() if types[key] is not None}
       
    def __call__(self, clip, n, lower, upper):        
        set_out = dict()
        for key, value in self.set1.items():
            if self.types[key] is not None:
                set_out[key] = value - ((n-lower)/(upper-lower) * self.diff[key])
                set_out[key] = round(set_out[key],2) if self.types[key] == float else round(set_out[key])
            else:
                set_out[key] = value 
        return self.function(clip, **set_out)


def distribute(n, clip, MAP, selection):
    if selection is not None:
        clip_master = clip
        clip = clip.std.CropAbs( *selection )
    iterator = iter(MAP)
    for (lower, upper) in iterator:
        funcs = next(iterator)
        if lower <= n <= upper:
            clip = functools.reduce(lambda r, f: f(r, n, lower, upper), funcs, clip)   #chaines filters: clip=f2(f1(clip)) etc.
    if selection is not None:
        return havsfunc.Overlay(clip_master, clip, x=selection[2], y=selection[3])
    else:
        return clip


def run(clip, MAP, selection=None, placeholder=None):
    if placeholder is None:
        placeholder = clip
    return core.std.FrameEval(placeholder, functools.partial(distribute, clip=clip, MAP=MAP, selection=selection))


if __name__ == '__main__':

    '''
    Usage:
    '''

    clip = core.std.BlankClip(color=(255,0,0), length=300)
    
    def data1(clip, n, lower, upper): return clip.text.Text(f'frame: {n}   interval to print: {lower} to {upper}',alignment=7)   
    def data2(clip, *args):           return clip.text.Text(f'filters can be chained', alignment=4)   
    def data3(clip, *args):           return clip.text.Text('Text that is fade in and out', alignment=1)
    
    def headline1(clip, *args):       return clip.text.Text('Our Headline')
    def headline2(clip, *args):       return clip.text.Text('... and other headline')
    def headline3(clip, *args):       return clip.text.Text('... third Headline')

    '''
    MAP is a list with pairs, first item is frame interval, second is a list of functions(filters)
    Frame intervals can overlap!
    
    Filters could be chained in three ways using MAP:
        #1.Chained within a line in a MAP:
            (0, clip.num_frames-1),     [denoise, blur],  #or more filters
            
        #2.separate in lines:
            (0, clip.num_frames-1),     [denoise],
            (0, clip.num_frames-1),     [blur],
            #but why, it could be done as above
            
        #3.Defining a function yourself:
            def my_filters((clip, *args):
                clip = havsfunc.QTGMC(clip, InputType=1, Preset="Fast", EZDenoise=2.0)
                return clip.std.BoxBlur()
                
            #then placing it in a MAP:
            (0, clip.num_frames-1),     [my_filters],
            
        #4.Defining selection to apply MAP filtering on clip
           It is defined when calling animate.run()
           clip = animate.run(clip, MAP, selection=(300,200,50,90)) #(width, height, left, top)
    '''

    end = clip.num_frames-1
     
    MAP = [
                 (0,   60),     [CrossfadeFromColor( (0,0,0) ), Crossfade(None, headline1)],
                 (60, 100),     [headline1],
                 (101,150),     [Crossfade(headline1, headline2)],
                 (151,200),     [headline2],
                 (201,250),     [Crossfade(headline2,headline3)],
                 (251, end),    [headline3],
                 (end-29,end),  [CrossfadeToColor( (0,0,0) )],       
                 (0, end) ,     [
                     #example defining functions using lambda:
                     lambda clip, *args: clip.text.Text('this is always visible, because filter is positioned on the bottom in MAP', alignment=4),
                     lambda clip, n, *args: clip.text.Text(f'frame: {n}', alignment=1),
                                ]
           ]
    clip = run(clip, MAP)
    #Or try selection opt. , same arguments as vapoursynth CropAbs(width=300,height=200,left=50,top=90)
    #clip = run(clip, MAP, selection=(300,200,50,90))
    clip.set_output()

