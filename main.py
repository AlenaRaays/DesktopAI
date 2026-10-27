from DesktopAgent import DesktopAgentEnvironment

def main():
    print("Запуск агента")
    env = DesktopAgentEnvironment()
    print("Агент готов")

    env.step()

if __name__ == "__main__":
    main()