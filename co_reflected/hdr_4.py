def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    slack_penalty = max(0, -js)
    wait_factor = jwt / (1 + avgwt)
    urgency = (jcd - tnow) * 0.3 - (tnow - jat) * 0.2
    if js < 5:
        slack_adj = js * 0.5
    else:
        slack_adj = -js * 0.1
    return jnpt + slack_penalty + wait_factor + (jd - tnow) * 0.2 + urgency + slack_adj
