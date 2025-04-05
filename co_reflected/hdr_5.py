def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    urgency = (jcd - tnow) * 0.3 - (tnow - jat) * 0.2
    slack_adj = js * 0.5 if js < 5 else -js * 0.1
    priority = jw * 0.2
    if mrel > 5:
        urgency += 2
    return jnpt + urgency - priority + jro * 0.4 + slack_adj
