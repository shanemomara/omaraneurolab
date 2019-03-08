# NeuroChaT
An analysis toolset for Neuroscience

# TODO
1. Look at calc_border_distance. Seems to fail for very large pixel values.
2. Don't crash on loading if there is no cut file. (Extension - don't crash, print error messages instead)
3. Show firing map plots in centimetres instead of in pixels.
4. Print a message if you try to set to a non existant unit number.
5. Set the open dir for files to self.path = Qt.Core.QFileInfo(excel_file).path() - write an open file function in ui and just update this.
6. Look at the centre point for the isi_corr.
7. Related to the below, create a set class.
8. Create a summary function which opens a set file, checks for cut files, loops over units, and plots a nice summary figure.