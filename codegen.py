class Variable():
  def __init__(self, name, vtype, addition=None, initial=None):
    self._name = name
    self._vtype = vtype
    self._addition = addition or ''
    self._initial = initial or ''
    
  def __str__(self):
    return '{self._addition} {self._vtype} {self._name} {self._initial}'.format(self=self)

class Array():
  def __init__(self, name, vtype, shape, addition=None, initial=None):
    self.
  
class Function():
  def __init__(self, name, ret, args=None, addition=None):
    self._name = name
    self._ret = ret
    if isinstance(args, list):
      self._args = ', '.join(map(str, args))
    else:
      self._args = args or 'void'
    self._addition = addition or ''
    
  def __str__(self):
    return '{i._addition} {i._ret} {i._name}({i._args})'.format(i=self)
    
if __name__ == '__main__':
  argc = Variable('argc', 'int')
  argv = Variable('argv', 'char**')
  func = Function('main', 'int', [argc, argv])
  print(func)