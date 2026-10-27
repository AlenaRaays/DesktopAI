import mss
import cv2
import numpy as np
import Model as md
import ActionM as am

class DesktopAgentEnvironment:

    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.input_dim = 32 * 32

        self.agent = md.TinyNumPyAgent(input_dim=self.input_dim, hidden_dim=64, output_dim=3)

    def get_state_vector(self) -> np.ndarray:
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        resized = cv2.resize(gray, (32, 32))
        vector = resized.astype(np.float32) / 255.0
        return vector.reshape(1, -1)
    
    def step(self):
        state = self.get_state_vector()

        action_id = self.agent.predict_action(state)
        print(f"Выбрано действие ID: {action_id}")

        if action_id == 0:
            am.MouseController.click_pkm()  # Вызов статического метода через класс
        elif action_id == 1:
            print("Действие 1: Нажатие стрелок для сортировки")
            am.pag.press('down')
            am.pag.press('down')
            am.pag.press('enter')
            am.pag.press('enter')
        else:
            print("Действие 2: Ожидание/Бездействие")
            am.time.sleep(1)

if __name__ == "__main__":
    env = DesktopAgentEnvironment()
    print("Агент готов")
    env.step()
