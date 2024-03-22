import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os



# Загрузить модель YOLO
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


# Функция для детекции классов с помощью модели
def predict_image(model, image, conf_threshold, iou_threshold):
    # Детектировать класс объекта, применив модель
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    # Посчитать сколько раз встретился каждый класс
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Создать текст с результатом детекции класса с помощью модели
    prediction_text = 'Результат детекции: '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "Заданных классов объектов не обнаружено"

    # Оценить время получения результата детекции класса объекта
    latency = sum(res[0].speed.values())  # in ms, need to convert to seconds
    latency = round(latency / 1000, 2)
    prediction_text += f' , время детекции: {latency} секунд.'

    # Преобразовать изображение в RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text


def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Детектор светящихся объектов",
        initial_sidebar_state="collapsed",
    )

    # Sidebar information
    st.sidebar.markdown("Автор: Семенов Артём")

    st.sidebar.markdown("Заказчик: ЦИТМ Экспонента")

    # Set custom CSS styles
    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Описание проекта
    st.markdown("<div class='title'>Детектор светящихся объектов</div>", unsafe_allow_html=True)

    # Добавить разделитель

    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Оценить эффективность</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Добавить разделитель
    st.markdown("---")

    # Добавить разделитель
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Выберите модель", ("Пожар", "Базовая"), index=0)

    models_dir = "general-models" if model_type == "Базовая" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    with col2:
        selected_model = st.selectbox("Выберите размер модели", sorted(model_files), index=1)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")  # type: ignore
    model = load_model(model_path)

    # Add a section divider
    st.markdown("---")

    # Set confidence and IOU thresholds
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Точность детекции", 0.0, 1.0, 0.20, 0.05)

    with col1:
        iou_threshold = st.slider("Степень пересечения между двумя ограничивающими рамками", 0.0, 1.0, 0.5, 0.05)

    # Add a section divider
    st.markdown("---")

    # Image selection
    image = None
    image_source = st.radio("Выберите источник данных:", ("Введите URL", "Загрузить с ПК"))
    if image_source == "Загрузить с ПК":
        # File uploader for image
        uploaded_file = st.file_uploader("Выбрать изображение", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        # Input box for image URL
        url = st.text_input("Введите URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Ошибка загрузки URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка загрузки URL: {e}")
                image = None

    if image:
        # Display the uploaded image
        with st.spinner("Выполняется обнаружение"):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)

        prediction = Image.fromarray(prediction)

        # Create a BytesIO object to temporarily store the image data
        # image_buffer = io.BytesIO()

        # Save the image to the BytesIO object in PNG format
        # prediction.save(image_buffer, format='PNG')

        # Create a download button for the image
        #st.download_button(
            #label='Сохранить результат детекции',
            #data=image_buffer.getvalue(),
            #file_name='prediciton.png',
            #mime='image/png'
        #)


if __name__ == "__main__":
    main()
