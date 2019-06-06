import numpy as np

class DescensoGradiente:
#parametros del constructor:
#tetas: matriz con los valores iniciales de los pesos o tetas
#ejemplos: matriz que tiene como fila un ejemplo y como columna valores de atributos para
#ese ejemplo
#salidas: vector o arreglo que contiene los valores de salida de cada ejemplo
#tarifaAprendizaje: contiene el valor para el rango de aprendizaje.

      def __init__(self,tetas,ejemplos,salidas,tarifaAprendizaje):
          self.tetas=tetas
          self.ejemplos=ejemplos
          self.y=salidas
          self.tarifaAprendizaje=tarifaAprendizaje
          self.cantidad_atributos=len(tetas)
#funcion que determina el valor de la hipotesis para un conjunto de tetas y atributos especificos
#valores de las variables en ese ejemplo
      def __Hipotesis(self,tetas,atributos):
          return np.sum(tetas*np.array(atributos))

#realiza la sumatoria para un atributo o teta especifico, barriendo todos los ejemplos de entrenamiento
#(la sumatorio corresponde a la pendiente de la recta tangente en ese punto de teta(teta0,teta1) especificos)
#funcion= sumatoria[i,m]{(h(x^[i])-y^[i])*x^[i]}

      def __sumatoria(self,tetas,atributo_evaluar):
           costo=0
           for i in range(len(self.ejemplos)):
               h=self.__Hipotesis(tetas,self.ejemplos[i])
               costo=costo+((h-self.y[i])*self.ejemplos[i,atributo_evaluar])
           return costo
#funcion que nos permite actualizar cada valor de teta por separado
      def __actualizar_teta(self,teta_actualizar):
           #print(self.tarifaAprendizaje,self.tetas[teta_actualizar],self.sumatoria(tetas,teta_actualizar))
           pendiente=round((self.tarifaAprendizaje*self.__sumatoria(self.tetas,teta_actualizar)),10)
           print(pendiente)
           return round(self.tetas[teta_actualizar]-pendiente,10)
#descenderemos hasta que los valores de teta sean iguales a los de la iteracion anterior(la pendiente sea igual a 0)
      def descender(self):
          lista_cambios=[]
          iguales=False
          #hasta que el algoritmo converja
          while not iguales:
                #actualizaremos cada atributo o teta
                for i in range(self.cantidad_atributos):
                      #guardamos cada teta en la lista para despues convertirla a vector
                     lista_cambios.append(self.__actualizar_teta(i))
                nuevas_tetas=np.array(lista_cambios)
                #limpiamos la lista
                lista_cambios.clear()
                #comparamos si la tetas obtenidas son iguales a las anteriores, entonces se dice que el algoritmo ha convergido
                if np.array_equal(nuevas_tetas,self.tetas):
                   iguales=True
                #actualizamos los valores de teta
                self.tetas=nuevas_tetas
                
          return self.tetas

"""
print("sumatoria:",sumatoria(tetas,ejemplos,0,y))
print("actualizado:",actualizar_teta(0.2,0,tetas,ejemplos,y))
print(tetas[0])
"""
#matriz de tetas o pesos que contienen los valores iniciales para las tetas
tetas=np.array([1,2])

#ejemplos que le daremos al algoritmo de aprendizaje para que pueda aprender 
#la funcón que modela el comportamiento de los mismos, y así poder hacer predicciones
ejemplos=np.matrix([[1,2],[1,4],[1,6],[1,8],[1,10],[1,12]])

#resultados de los ejemplos o y´s
y=np.array([12,22,32,42,52,62])

#creamos un objeto de tipo DescensoGradiente pasando los parametros debidos 
DG=DescensoGradiente(tetas,ejemplos,y,0.003)
print(DG.descender())











    
