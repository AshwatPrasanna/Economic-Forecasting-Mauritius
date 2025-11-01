# MARK: Opening the CSV and cleaning the Data

def readData(fileName): # This returns a tuple with keys, then values
    file = open(fileName, "r")
    lines = file.readlines()
    keys = []
    values = []

    for line in lines:
        if "\n" in line:
            line = line[:-1]
        if "\ufeff" in line:
            line = line[1:]
            
        data = line.split(",")
        keys.append([float(string) for string in data[:-1]])
        values.append(float(data[-1]))

    return (keys, values)
            
