import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


class GFitLayout(GridLayout):
    def __init__(self, **kwargs):
        super(GFitLayout, self).__init__(**kwargs)

        self.cols = 2
        self.add_widget(Label(text="Name: "))
        self.name = TextInput(multiline=False)
        self.add_widget(self.name)

        self.add_widget(Label(text="Age: "))
        self.age = TextInput(multiline=False)
        self.add_widget(self.age)

        self.add_widget(Label(text="Height: "))
        self.height = TextInput(multiline=False)
        self.add_widget(self.height)

        self.add_widget(Label(text="Weight: "))
        self.weight = TextInput(multiline=False)
        self.add_widget(self.weight)


class GFit(App):
    def build(self):
        return GFitLayout()


if __name__ == '__main__':
    GFit().run()
