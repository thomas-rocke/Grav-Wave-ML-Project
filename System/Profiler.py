# Profiling wrapper to measure software performance
#Taken from https://osf.io/upav8/ & https://www.youtube.com/watch?v=8qEnExGLZfY

import cProfile, pstats, io



def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'tottime'#'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open ("profile.txt", 'w') as f:
            f.writelines(s.getvalue())
        #print(s.getvalue())
        return retval

    return inner