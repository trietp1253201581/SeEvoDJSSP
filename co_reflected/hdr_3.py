def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    dynamic_wait = jwt / (1 + avgwt)
    urgency = jcd - (tnow - jat) * 0.2
    slack_adj = js * 0.5 if js <5 else -js * 0.1
    return jrt * 0.5 + dynamic_wait * 0.3 + jro + dynamic_wait + urgency + slack_adj
