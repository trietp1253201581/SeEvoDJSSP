def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    score = jnpt + js - jwt * 0.1 + jro * 0.5
    urgency = jcd - (tnow - jat) * 0.2
    score += urgency
    return score
