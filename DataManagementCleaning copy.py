
import os.path
import shutil


EntireCovidSet = "/Users/amelianelson/Desktop/Imaging/manifest-1711845412213/COVID-19-NY-SBU"

#Goes through every patient directory downloaded
with os.scandir(EntireCovidSet) as Patients:
    for patient in Patients:
        picNum = 1
        if patient.name != ".DS_Store":
            if patient.name != "LICENSE":
                with os.scandir(patient.path) as Dates:
                    for date in Dates:
                        if date.name != ".DS_Store":
                            with os.scandir(date.path) as Pictures:
                                for picture in Pictures:
                                    if picture.name != ".DS_Store":
                                        os.rename(picture.path +"/1-1.dcm", picture.path +"/"+ patient.name +str(picNum) + ".dcm")
                                        shutil.copy(picture.path +"/"+ patient.name + str(picNum) +".dcm", "/Users/amelianelson/Desktop/UsefulImages")
                                        picNum += 1
                                        firstDate = False
                                        firstPicture = False
                shutil.rmtree(patient.path)


                
    
    





        