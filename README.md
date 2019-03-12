# NeuroChaT
A fork of an analysis toolset for Neuroscience.

## Things to do and considerations, not in priority order.
1. Look at calc_border_distance as it seems to fail for very large pixel values.
2. Don't crash, print error messages instead, when:
    - An Excel file is open that need to write to. (PermissionError).
    - When loading a spike and there is no cut file. (IOError)
    - When trying to save to a pdf which is already open (PermissionError).
4. Print a message if you try to set to a non existant unit number.
5. Set the open dir for files to self.path = Qt.Core.QFileInfo(excel_file).path() - write an open file function in ui and just update this.
7. Related to the below, create a set class.
8. Create a summary function which opens a set file, checks for cut files, loops over units, and plots a nice summary figure.
9. Plot spike-rasters as lines instead of image-style.
10. Add %refactory_violation to isi analysis output.
11. Get ROC measurements by comparing a Gaussian to the distribution of spike amplitudes.