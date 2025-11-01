# MARK: Opening the CSV and cleaning the Data

def readData(fileName): # This returns a tuple with keys, then values
    file = open(fileName, "r")
    lines = file.readlines()
    cleanedLines = []

    for line in lines:
        if "\n" in line:
            line = line[:-1]
        if "\ufeff" in line:
            line = line[1:]

        cleanedLines.append(line)
            
    data1 = cleanedLines[0].split(",")
    data2 = cleanedLines[1].split(",")
    mins = [round(float(x), 3) for x in data1[:-1]]
    maxes = [round(float(x), 3) for x in data2[:-1]]
    
    minref = round(float(data1[-1]), 3)
    maxref = round(float(data2[-1]), 3)

    return (mins, maxes, minref, maxref)
            

