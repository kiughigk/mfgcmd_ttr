"""
Description : Utilitities
"""

import os

def mkdir(dirName):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)


def convert_multi2single_column(df):

    oldColList = list(df.columns)
    newColList = []
    for col in oldColList:
        if (type(col) == tuple):
            newCol = str(col[0])
            if (len(col) > 1):
                for i in range(1, len(col)):
                    if (str(col[i]) != ''):
                        newCol += '_'+str(col[i])
            newColList.append(newCol)
        else:
            newColList.append(col)    

    df.columns = newColList

    return(df)

