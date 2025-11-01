def norm(x):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (((x[i]-minV)/(maxV-minV))*2.0)-1.0

    return x


