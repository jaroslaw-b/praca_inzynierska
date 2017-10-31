import visionAlgorithm



wizyjny = visionAlgorithm.VisionAlgorithm()
for nb_of_test_image in range(18, 24):
    wizyjny.load_image(nb_of_test_image)
    # wizyjny.show_image(wizyjny.image_to_display)
    wizyjny.process_image2()
    # wizyjny.show_image(wizyjny.image_to_display)

    # wizyjny.preprocessing()
    wizyjny.process_image()
    # wizyjny.search_damage()
    # wizyjny.show_image(wizyjny.image_to_display)
    wizyjny.show_image2()