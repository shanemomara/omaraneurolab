# NeuroChaT

A fork of an analysis toolset for Neuroscience.

## Things to do and considerations, not in priority order.

2. Don't crash, print error messages instead, when:
   - An Excel file is open that need to write to. (PermissionError).
   - When loading a spike and there is no cut file. (IOError)
   - When trying to save to a pdf which is already open (PermissionError).
3. Print a message if you try to set to a non existant unit number.
4. Set the open dir for files to self.path = Qt.Core.QFileInfo(excel_file).path() - write an open file function in ui and just update this.
5. Plot spike-rasters as lines instead of image-style.
6. Get ROC measurements by comparing a Gaussian to the distribution of spike amplitudes.

## Ideas

1. Cache the non moving periods of the subjects movement to avoid many calculations
2. Polynomial regression on phase vs position plots/
