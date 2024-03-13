import cv2

def crop_image():
    img = cv2.imread("C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/tfg/SLP/danaLab/00006/IR/cover1/image_000005.png")
    cv2.imshow('hola',img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    original_height, original_width = img.shape[:2]
    print(original_height,original_width)
    new_height, new_width = 84, 192

    start_x = (original_width // 2) - (new_width//2)
    start_y = (original_height // 2) - (new_height//2)
    end_x = start_x + new_width
    end_y = start_y + new_height

    # Recortar la imagen
    cropped_image = img[start_y:end_y, start_x:end_x]

    # Mostrar y guardar la imagen recortada
    cv2.imshow('Imagen Recortada', cropped_image)
    #cv2.imwrite(os.path.join(IMG_PATH,f"{module}_image_{random_patient}.jpg"), img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

crop_image()