'''<b>ExportCroppedObjects</b> crop each object to its bounding box and export as an individual image file.
<hr>
Specify an object set, an image, and an output folder.
For each object of the set, the bounding box of the object will be determined, the image will be cropped to the bounding box and a corrsponding image file is written to the output folder. 

Author: Volker.Hilsenstein@embl.de

'''
#################################
# Imports from useful Python libraries
#################################

import numpy as np
import scipy.ndimage as scind

#################################
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
##################################
import pdb
import sys
import os
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp

import PIL.Image as PILImage

##################################
# Constants
###################################

# Rescaling options
RS_MINMAX = "Use min/max intensity (fit range)"
RS_1 =   "Don't rescale (multiply by 1.0)"
RS_255 = "Multiply by 255.0"
RS_212 = "Mutliply by 2^12-1"
RS_216 = "Mutliply by 2^16-1"

OUTFOLDER_INPUT = "Same as input image"
OUTFOLDER_DEFAULT = "Default output folder"
OUTFOLDER_CUSTOM = "Specify folder"

BITDEPTH8 = "8 bpp"
BITDEPTH16 = "16 bpp"

####

class ExportCroppedObjects(cpm.CPModule):

    ################### Name ##########################
    module_name = "ExportCroppedObjects"
    category = "Object Processing"
    variable_revision_number = 1
    
    ################# GUI Settings ########################
    def create_settings(self):
        self.input_object = cps.ObjectNameSubscriber("Object set:",doc = """Pick the objects you want to export.""")

        self.input_image = cps.ImageNameSubscriber("Select the image to measure", doc = '''Image''')
        self.rescale = cps.Choice("How to rescale the image",
                                      [RS_MINMAX,RS_255],RS_MINMAX,doc="""
                                      <i>Select the rescaling method</i><br>
                                      the following options are possible: map maximum intensity to max destination range, multiply intensity by 255, multiply by 2^12, multiply by 2^16""")

        self.outfolder_choice = cps.Choice("Output folder location",
                                      [OUTFOLDER_INPUT,OUTFOLDER_DEFAULT, OUTFOLDER_CUSTOM ],OUTFOLDER_INPUT,doc="""
                                      Determine where the cropped object images are to be saved""")


        self.padbbox = cps.Integer("Extend bounding box by how many pixels:",0, """With the default value of 0, the image is cropped tightly to the bounding box of each object. By specifying integers > 0, the bounding box is enlarged by the specified number of pixels on each side, thus including more image context around the object. The enlarged bounding box will be clipped such that it lies entirely in the image""") 

        self.outfolder_path = cps.Text("Output folder path", os.path.expanduser("~"),
                                       metadata = True,
                                        doc="""
                                        Put your output path here""")
        
        
    def settings(self):
        #return  [self.input_object, self.input_image, self.rescale,self.padbbox, self.single_file_name]
        return  [self.input_object, self.input_image, self.rescale, self.outfolder_choice ,self.outfolder_path ]

    def visible_settings(self):
        if self.outfolder_choice.value == OUTFOLDER_CUSTOM:
            return self.settings()
        else:
            tmp = self.settings()
            tmp.remove(self.outfolder_path)
            return tmp

    # Main
    def run(self, workspace):


        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)


        image = workspace.image_set.get_image(self.input_image.value)
        objects = workspace.get_objects(self.input_object.value)
        assert isinstance(objects, cpo.Objects)
        indices = objects.get_indices()
        labeled = objects.segmented
        bboxes = []
        for i in indices: # go through all object labels 
            # get coordinates of all pixels belonging to object
            pixelpos = np.argwhere(labeled == i)
            # create a bounding box as in http://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in
            (ystart, xstart), (ystop, xstop) = pixelpos.min(0), pixelpos.max(0) + 1 
            bbox=(ystart,ystop,xstart,xstop)
            bboxes.append((i, bbox))

        self.save_crop_image(workspace,bboxes)
            
    def is_interactive(self):
        return False
    

    def save_crop_image(self, workspace, bboxes):
        # we simplify things compared to the saveimages module by assuming we have libtiff ...
        # we don't support colormaps for greyscale images

        image = workspace.image_set.get_image(self.input_image.value)
        origfilename = os.path.split(image.get_file_name())[-1]
        basefilename = os.path.splitext(origfilename)[0]

        if self.outfolder_choice.value == OUTFOLDER_CUSTOM:
            basepath = self.outfolder_path.value
        elif self.outfolder_choice.value == OUTFOLDER_INPUT:
            fullfilepath = image.get_path_name() + os.sep + image.get_file_name() # this step may appear superfluous but I noticed that image.get_file_name() sometimes still contains subfolders
            basepath = os.path.split(fullfilepath)[0]
        else:
            basepath = cpp.get_default_image_directory()


        pixels = image.pixel_data
        if self.rescale.value == RS_MINMAX:
            pixels = pixels.copy()
            # Normalize intensities for each channel
            if pixels.ndim == 3:
                # RGB
                for i in range(3):
                    img_min = np.min(pixels[:,:,i])
                    img_max = np.max(pixels[:,:,i])
                    if img_max > img_min:
                        pixels[:,:,i] = (pixels[:,:,i] - img_min) / (img_max - img_min)
            else:
                # Grayscale
                img_min = np.min(pixels)
                img_max = np.max(pixels)
                if img_max > img_min:
                    pixels = (pixels - img_min) / (img_max - img_min)
        else:
            # Clip at 0 and 1
            if np.max(pixels) > 1 or np.min(pixels) < 0:
                sys.stderr.write(
                    "Warning, clipping image %s before output. Some intensities are outside of range 0-1" %
                    self.image_name.value)
                pixels = pixels.copy()
                pixels[pixels < 0] = 0
                pixels[pixels > 1] = 1
                        
        pixels = (pixels*255).astype(np.uint8)
            
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            mode = 'RGBA'
        elif pixels.ndim == 3:
            mode = 'RGB'
        else:
            mode = 'L'

        #filename = self.get_filename(workspace)
        # TODO: create filename
        for box in bboxes:
            limits =  box[1]
            crop = pixels[limits[0]:limits[1], limits[2]:limits[3]]

            # enlarge bounding boxes, if the user wants to
            if self.padbbox.value !=0:
                print "Bounding box padding not implelemted yet" # TODO
                pass # do something
            
            pil = PILImage.fromarray(crop,mode)
            filename = basepath + os.sep + basefilename + "_object_"+str(box[0]).zfill(len(str(len(bboxes))))+".tif"
            pil.save(filename)
