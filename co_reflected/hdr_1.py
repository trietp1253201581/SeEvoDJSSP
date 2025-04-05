def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    urgency = (jcd - tnow) * 0.3 + (jd - tnow) * 0.2
    slack_penalty = max(0, 5 - js) * 0.5
    dynamic_wait = jwt / (1 + avgwt) * 0.2
    return jnpt + js * 0.2 + urgency + jro * 0.5 - dynamic_wait - jwt * 0.1 + slack_penalty
