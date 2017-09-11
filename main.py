import visionAlgorithm

wizyjny = visionAlgorithm.VisionAlgorithm()
wizyjny.load_image()
wizyjny.analyse_image()
wizyjny.show_image(wizyjny.image_resized)
wizyjny.show_image(wizyjny.image_temp)
