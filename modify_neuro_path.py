import os
import shutil

# Modifies the first occurence of sys path to current working dir
def main():
    here = os.getcwd()
    file_loc = os.path.join("neurochat_gui", "neurochat_ui.py")
    neuro_file = open(file_loc, "r")
    temp_file = open("temp.txt", "w")
    for line in neuro_file:
        if "sys.path.insert" in line:
            line = "sys.path.insert(1, r\'" + here + "\')"
            print("Set the path to: " + line)
            temp_file.write(line)
            temp_file.write("\n")
        else:
            temp_file.write(line)
    neuro_file.close()
    temp_file.close()

    os.remove(file_loc)
    shutil.move("temp.txt", file_loc)

if __name__ == "__main__":
    main()