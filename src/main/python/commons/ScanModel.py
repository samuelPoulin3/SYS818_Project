#! /usr/bin/env python3

DEBUG = False

class ScanModel():
    """
        The model of a scan. Contains the informations for a scan. 

        Properties:
            scan_number: String
                The number of a scan
            scan_type: String
               The type of a scan
            vox_res: String
                The resolution of a scan in mm. ex 0.0 x 0.0 x 0.00 
            rect_fov: String
                ? ex 000/000
            orientation: String
                Orientation of the picture. Valid values are:
                Sag: sagital
                ...
            tr: Float
                ? in ms.
            te: Float
                ? in ms.
            ti: Float
                ? in ms.
            flip: Int
                ?
            image:

            pixels:



            
    """
    __slots__ = ('scan_number', 'scan_type', 'vox_res', 'rect_fov', 'orientation', 'tr', 'te', 'ti', 'flip', 'image', 'pixels')
    def __init__(self, *args, **kwargs):
        self.scan_number = None
        self.scan_type = None
        self.vox_res = None
        self.rect_fov = None
        self.orientation = None
        self.tr = None
        self.te = None
        self.ti = None
        self.flip = None
        self.image = None
        self.pixels = None
