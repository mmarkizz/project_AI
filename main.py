import speech_recognition as sr

def simple_surname_recognizer():
    r = sr.Recognizer()    # объект распознаватель который будет обрабатывать аудио
    
    with sr.Microphone() as source:
        print("Скажите фамилию...")
        r.adjust_for_ambient_noise(source)    #анализ фонового шума
        audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio, language='ru-RU')  #Получаем распознанный текст с помощью гугл спич апи
        surname = text.strip().title()
        print(f"Распознанная фамилия: {surname}")
        
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса: {e}")

# Использование
if __name__ == "__main__":
    simple_surname_recognizer()