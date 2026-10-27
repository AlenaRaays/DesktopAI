import pyautogui as pag
import time

pag.PAUSE = 0.5
pag.FAILSAFE = True

class MouseController:
    @staticmethod
    def move_to(x: int, y: int, duration: float = 0.3):
        pag.moveTo(x, y, duration=duration)

    @staticmethod
    def click_lkm(x: int = None, y: int = None):
        if x is not None and y is not None:
            pag.click(x, y, button="left")
        else:
            pag.click(button="left")

    @staticmethod
    def click_pkm(x: int = None, y: int = None):
        if x is not None and y is not None:
            pag.click(x, y, button="right")
        else:
            pag.click(button="right")

    @staticmethod
    def down():
        pag.press('down')

    @staticmethod
    def enter():
        pag.press('enter')