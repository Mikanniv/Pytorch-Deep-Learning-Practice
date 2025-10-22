class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        print(f'{self.name} makes a sound.')

class Dog(Animal):
    def __call__(self, *args, **kwargs):
        print(f'{self.name} eats a {args[0]}.')

dog = Dog('jhd')
dog.speak()
dog('apple')