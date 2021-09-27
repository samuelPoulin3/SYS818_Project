import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
import config
from MainWindow import MainWindow

class ClassificationAlgorithm(ApplicationContext):
    def run(self):
        config.app_context = self
        window = MainWindow()
        version = self.build_settings['version']
        window.setWindowTitle("Classification algorithm - V" + version)
        window.resize(500, 100)
        window.show()
        return self.app.exec_()

def main():
    app = ClassificationAlgorithm()
    sys.exit(app.run())


main()