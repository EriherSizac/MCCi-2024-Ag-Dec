import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Carga y Manipulación de la Imagen

# Capturar una fotografía usando la cámara web
def capture_image_from_webcam():
    # Inicializa la cámara (0 es el índice de la cámara predeterminada)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return None

    # Lee un frame de la cámara
    ret, frame = cap.read()

    # Libera la cámara
    cap.release()

    if not ret:
        print("Error: No se pudo capturar la imagen.")
        return None

    # Guardar la imagen capturada
    cv2.imwrite('captured_image.jpg', frame)

    # Mostrar la imagen capturada
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return frame

# Llamar a la función para capturar la imagen
image = capture_image_from_webcam()

# Verificar si la imagen se capturó correctamente
if image is None:
    print("Error: No se pudo capturar la imagen. Verifica la cámara.")
    exit()


# Intercambiar los canales verde y azul
# Cambia el canal verde con el azul
swapped_image = image.copy()
swapped_image[:, :, [0, 1]] = swapped_image[:, :, [1, 0]]

# Convertir a escala de grises
# Convierte la imagen original a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convertir a HSV
# Convierte la imagen original al espacio de color HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Mostrar las imágenes
cv2.imshow('Original Image', image)
cv2.imshow('Swapped Channels', swapped_image)
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Captura de Video y Estimación del Número F

def estimate_f_number(camera_index=0, reference_width=5.0, known_distance=30.0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return None, None

    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el video.")
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Usar detección de bordes o segmentación para medir el ancho en píxeles del objeto
    # Aquí necesitarías un algoritmo de detección de bordes o un método para medir píxeles
    # por simplicidad asumamos que medimos manualmente por ahora:
    pixels_width = 50  # Ejemplo de valor

    # Estimar el número f usando la fórmula básica
    focal_length = (pixels_width * known_distance) / reference_width
    print(f"Número f estimado: {focal_length}")

    cap.release()
    cv2.destroyAllWindows()
    return focal_length, pixels_width

# Definir el ancho del objeto de referencia y las distancias conocidas
reference_width = 5.0  # en cm
distances = [30.0, 50.0, 70.0]  # Distancias de ejemplo en cm
errors = []

for d in distances:
    estimated_f, pixels_width = estimate_f_number(reference_width=reference_width, known_distance=d)
    if estimated_f is None or pixels_width is None:
        continue
    estimated_distance = (reference_width * estimated_f) / pixels_width
    error = abs(estimated_distance - d)
    errors.append(error)
    print(f"Distancia estimada: {estimated_distance}, Error: {error}")

# Reportar el error
mean_error = np.mean(errors)
print(f"Error promedio en la medición: {mean_error}")

# 3. Mejoras de la Imagen

# Ecualización de histograma
# Aplica la ecualización de histograma a la imagen en escala de grises
equalized_image = cv2.equalizeHist(gray_image)

# Ecualización de canales independientemente
# Aplica la ecualización de histograma a cada canal de color por separado
equalized_channels = []
for i in range(3):
    equalized_channels.append(cv2.equalizeHist(image[:, :, i]))
equalized_image_rgb = cv2.merge(equalized_channels)

# Ecualización en el canal V de la imagen HSV
# Aplica la ecualización de histograma al canal V de la imagen HSV
hsv_equalized = hsv_image.copy()
hsv_equalized[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
hsv_to_rgb = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

# Aplicar realce Gamma
# Aplica corrección gamma para realzar áreas de baja y alta intensidad
gamma = 2.2
gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# Mostrar las imágenes
cv2.imshow('Equalized Image', equalized_image)
cv2.imshow('Equalized RGB Image', equalized_image_rgb)
cv2.imshow('HSV to RGB after V equalization', hsv_to_rgb)
cv2.imshow('Gamma Corrected Image', gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Representación Piramidal

# Pirámide de reducción
# Realiza la reducción de la imagen en niveles de la pirámide
def pyramid_downsample(image, levels):
    pyr_images = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyr_images.append(image)
    return pyr_images

pyramid_images = pyramid_downsample(image, 4)

# Upsample usando interpolación bilineal
# Usamos el último nivel de la pirámide para reescalar a tamaño original
upsampled_image_bilinear = cv2.resize(pyramid_images[-1], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

# Mostrar las imágenes de la pirámide
for idx, img in enumerate(pyramid_images):
    cv2.imshow(f'Pyramid Level {idx}', img)

cv2.imshow('Upsampled Image Bilinear', upsampled_image_bilinear)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Aplicación de Ruido y Filtrado

# Añadir ruido gaussiano
# Añade ruido gaussiano a la imagen
noise = np.zeros(image.shape, np.uint8)
cv2.randn(noise, 0, 25)  # media y desviación estándar
noisy_image = cv2.add(image, noise)

# Filtro de media, mediana, anisotrópico
# Aplica varios tipos de filtros a la imagen con ruido
mean_filtered = cv2.blur(noisy_image, (5, 5))
median_filtered = cv2.medianBlur(noisy_image, 5)
# Anisotrópico - Usamos un filtro bilateral como aproximación
aniso_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)

# Ruido sal y pimienta
# Añade ruido de sal y pimienta a la imagen
salt_pepper_noise = np.copy(image)
prob = 0.02
# Sal
salt_pepper_noise[np.random.rand(*image.shape[:2]) < prob] = 255
# Pimienta
salt_pepper_noise[np.random.rand(*image.shape[:2]) < prob] = 0

# Mostrar las imágenes
cv2.imshow('Gaussian Noise', noisy_image)
cv2.imshow('Mean Filter', mean_filtered)
cv2.imshow('Median Filter', median_filtered)
cv2.imshow('Anisotropic Filter', aniso_filtered)
cv2.imshow('Salt and Pepper Noise', salt_pepper_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Detección de Bordes

# Gradiente de la imagen en escala de grises
# Calcula el gradiente de la imagen en escala de grises
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
gradient_angle = np.arctan2(sobely, sobelx)

# Mostrar en canales RGB como se describe
# Muestra el gradiente en
