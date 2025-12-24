import torch
import math
import numpy as np
import random as rnd
from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Parameter
import matplotlib.pyplot as plt
from svetlanna import wavefront as w
from svetlanna.units import ureg
from svetlanna import LinearOpticalSetup
from svetlanna.elements import Element
from svetlanna.simulation_parameters import SimulationParameters
from svetlanna.parameters import OptimizableFloat
from svetlanna.wavefront import Wavefront, mul
from svetlanna.axes_math import tensor_dot
from typing import Iterable
from svetlanna.specs import PrettyReprRepr, ParameterSpecs
from svetlanna.visualization import jinja_env, ElementHTML
import gc
import signal
import sys
#----------------Lens Declaration---------------

class CylindricalLens(Element):
    """A class that describes the field after propagating through a
    cylindrical thin lens.
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: OptimizableFloat,
        radius: float = torch.inf,
        axis: str = 'x'  # Добавляем параметр оси фокусировки
    ):
        super().__init__(simulation_parameters)

        self.focal_length = self.process_parameter(
            'focal_length', focal_length
        )
        self.radius = self.process_parameter(
            'radius', radius
        )
        self.axis = axis  # 'x' или 'y'

        # Compute wave_number как в оригинальном коде
        wave_number, axes = tensor_dot(
            2 * torch.pi / self.simulation_parameters.axes.wavelength,
            torch.tensor([[1]], device=self.simulation_parameters.device),
            'wavelength',
            ('H', 'W')
        )
        self._wave_number = self.make_buffer('_wave_number', wave_number)
        self._calc_axes = axes

        # Создаем отдельные сетки для x и y
        x_linear = self.simulation_parameters.axes.W
        y_linear = self.simulation_parameters.axes.H

        # Сохраняем отдельные компоненты для цилиндрической линзы
        self._x_grid_sq = self.make_buffer(
            '_x_grid_sq',
            x_linear[None, :]**2  # shape: (1, 'W'), уже возведено в квадрат
        )
        
        self._y_grid_sq = self.make_buffer(
            '_y_grid_sq', 
            y_linear[:, None]**2  # shape: ('H', 1), уже возведено в квадрат
        )

        # Апертурная маска (используем общий радиус)
        self._radius_squared = self.make_buffer(
            '_radius_squared',
            self._x_grid_sq + self._y_grid_sq  # x² + y²
        )
        
        if self.radius == torch.inf:
            self._radius_mask = 1.0
        else:
            self._radius_mask = self.make_buffer(
                '_radius_mask',
                (self._radius_squared <= self.radius**2).to(
                    dtype=torch.get_default_dtype()
                )
            )

    @property
    def transmission_function(self) -> torch.Tensor:
        if self.axis == 'x':
            # Цилиндрическая линза, фокусирующая по оси X
            phase_factor = self._x_grid_sq
        elif self.axis == 'y':
            # Цилиндрическая линза, фокусирующая по оси Y  
            phase_factor = self._y_grid_sq
        else:
            raise ValueError("axis must be 'x' or 'y'")

        return torch.exp(
            -1j * self._radius_mask * phase_factor * (
                self._wave_number / (2 * self.focal_length)
            )
        )

    def get_transmission_function(self) -> torch.Tensor:
        """Returns the transmission function of the cylindrical lens."""
        return self.transmission_function

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Calculates the field after propagation through the cylindrical lens."""
        return mul(
            incident_wavefront,
            self.transmission_function,
            self._calc_axes,
            self.simulation_parameters
        )

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        """Calculates the field after passing through the lens during back propagation."""
        return mul(
            transmission_wavefront,
            torch.conj(self.transmission_function),
            self._calc_axes,
            self.simulation_parameters
        )

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                'focal_length', [
                    PrettyReprRepr(self.focal_length),
                ]
            ),
            ParameterSpecs(
                'radius', [
                    PrettyReprRepr(self.radius)
                ]
            ),
            ParameterSpecs(
                'axis', [
                    PrettyReprRepr(self.axis)
                ]
            )
        ]

    @staticmethod
    def _widget_html_(
        index: int,
        name: str,
        element_type: str | None,
        subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template('widget_lens.html.jinja').render(
            index=index, name=name, subelements=subelements
        )

#----------------Net Parameters-----------------

wavelength = 1064 * ureg.nm # wavelength, mm
lx = 30 * ureg.mm # screen size along x-axis, mm
ly = 16 * ureg.mm # screen size along y-axis, mm

Nx = 8192 # number of nodes along x-axis
Ny = 8192 # number of nodes along y-axis

x_length = torch.linspace(-lx / 2, lx / 2, Nx)
y_length = torch.linspace(-ly / 2, ly / 2, Ny)

#----------------define simulation parameters---
params = SimulationParameters(
    axes={
            'W': x_length,
            'H': y_length,
            'wavelength': wavelength
        }
)
x_grid, y_grid = params.meshgrid(x_axis='W', y_axis='H')

#----------------SLM Size----------------------
slm_height = 8 * ureg.mm
slm_width = 15 * ureg.mm
#----------------About slm---------------------
mac_size = 20
y_res = 800
x_res = 1500

x_mac_size = x_res//mac_size
y_mac_size = y_res//mac_size

#--------------SLM for aperture retrieval------
mask = torch.zeros(y_res, x_res)

slm = elements.SpatialLightModulator(
    simulation_parameters=params,
    mask=mask,
    height=slm_height,
    width=slm_width
)
slm_aperture = slm.get_aperture
slm_transmission_function = slm.transmission_function

#----------------For cheating version only: finding coordinates

def calculate_macro_coordinates():
    """Единая функция для расчета координат макропикселей"""
    y_indices, x_indices = torch.where(slm_aperture == 1)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    total_width = x_max - x_min + 1
    total_height = y_max - y_min + 1
    
    # Точные размеры макропикселей
    macro_width = total_width / x_mac_size
    macro_height = total_height / y_mac_size
    
    # Списки границ для каждого макропикселя
    x_bounds = []
    y_bounds = []
    
    for i in range(x_mac_size):
        start_x = x_min + int(i * macro_width)
        end_x = x_min + int((i + 1) * macro_width) if i < x_mac_size - 1 else x_max + 1
        x_bounds.append((start_x, end_x))
    
    for k in range(y_mac_size):
        start_y = y_min + int(k * macro_height)
        end_y = y_min + int((k + 1) * macro_height) if k < y_mac_size - 1 else y_max + 1
        y_bounds.append((start_y, end_y))
    
    return x_bounds, y_bounds, x_min, x_max, y_min, y_max

x_bounds, y_bounds, x_min, x_max, y_min, y_max = calculate_macro_coordinates()

#---------------Field choice:------------------
incident_wave = w.Wavefront.plane_wave(
   simulation_parameters=params,
   distance=1 * ureg.cm,
   wave_direction=[0, 0, 1]
)

#---------------Incident vector:------------------
def vector_gen(kind = None, discretisation = False, x = None, frequency = None, same_as100 = False):
    vector = torch.zeros(x_mac_size)
    
    if kind == None:
        vector = torch.rand(x_mac_size)
    elif kind == "ones":
        vector = torch.ones(x_mac_size)
    elif kind == "sin":
        for i in range(x_mac_size):
            if frequency == None: frequency = 0.1
            vector[i] = abs(np.sin(frequency*i))
    elif kind == "cos":
        for i in range(x_mac_size):
            
            vector[i] = abs(np.cos(frequency*i))
    elif kind == "rand":
        vector = x*torch.rand(x_mac_size)
    elif kind == "x_rand":
        vector = x+(1-x)*torch.rand(x_mac_size)
    if same_as100 == True:
        vector[15:] = 0
    target = vector

    if discretisation == True:
        target = torch.zeros(x_mac_size//2)
        vector[::2] = 0
        target = vector[1::2]
    
    return vector, target

vect, target_v = vector_gen("ones", True, 0.1, same_as100=True)
print("Vector:", vect)

def incident_amplitude(vector=None, reference=False, discretisation=False):
    mask_copy = torch.clone(slm_aperture)
    
    if vector is None:
        vector = torch.rand(x_mac_size)

    for i in range(len(vector)):
        start_x, end_x = x_bounds[i]
        mask_copy[:, start_x:end_x] = mask_copy[:, start_x:end_x] * vector[i]

    # Референсный ряд - ИСПРАВЛЕНА ОШИБКА ТИПА
    if reference == True:
        # Используем первую строку y_bounds для определения высоты референсного ряда
        ref_start_y, ref_end_y = y_bounds[0]
        ref_height = ref_end_y - ref_start_y
        mask_copy[y_min:y_min+ref_height, x_min:x_max] = 1

    # Горизонтальные линии - ИСПРАВЛЕНА ОШИБКА: используем точные границы из y_bounds
    if discretisation == True:
        for i in range(y_mac_size):
            start_y, end_y = y_bounds[i]
            if i % 2 == 0:
                mask_copy[start_y:end_y, x_min:x_max] = 0
                
    return mask_copy
inc_ampl = incident_amplitude(vect, False, True)

def shift_inc_ampl_matrix(inc_ampl, x_bounds, y_bounds, shift_down=0, shift_left=0):
    """
    Смещает матрицу inc_ampl вниз и влево на заданное количество пикселей.
    
    Parameters:
    -----------
    inc_ampl : torch.Tensor
        Исходная матрица амплитуд
    x_bounds : list
        Границы макропикселей по оси X
    y_bounds : list
        Границы макропикселей по оси Y  
    shift_down : int
        Количество пикселей для смещения вниз (в единицах SLM пикселей)
    shift_left : int
        Количество пикселей для смещения влево (в единицах SLM пикселей)
    
    Returns:
    --------
    torch.Tensor
        Смещенная матрица амплитуд
    """
    
    # 1. Определяем средний размер макропикселя
    if len(x_bounds) > 0 and len(y_bounds) > 0:
        # Берем первый макропиксель для определения размеров
        first_x_start, first_x_end = x_bounds[0]
        first_y_start, first_y_end = y_bounds[0]
        
        macro_width_pixels = first_x_end - first_x_start
        macro_height_pixels = first_y_end - first_y_start
        
        # 2. Вычисляем коэффициент масштабирования между SLM пикселями и общей сеткой
        scale_x = macro_width_pixels // mac_size
        scale_y = macro_height_pixels // mac_size
        ratio_x = shift_left/macro_width_pixels*100
        ratio_y = shift_down/macro_height_pixels*100
        print(f"Размер макропикселя в общей сетке: {macro_width_pixels}x{macro_height_pixels}")
        
        print(f"Масштабный коэффициент: {scale_x}x{scale_y}")
        
        # 3. Преобразуем смещение из SLM пикселей в пиксели общей сетки
        shift_down_pixels = shift_down * scale_y
        shift_left_pixels = shift_left * scale_x
        
        print(f"Смещение: вниз на {shift_down_pixels} пикс., влево на {shift_left_pixels} пикс.")
        
        # 4. Создаем новую матрицу с таким же размером как inc_ampl
        shifted_inc_ampl = torch.zeros_like(inc_ampl)
        
        # 5. Вычисляем новые границы после смещения
        height, width = inc_ampl.shape
        
        # Определяем область, которая останется после смещения
        start_y = shift_down_pixels
        end_y = height
        start_x = shift_left_pixels  
        end_x = width
        
        # Определяем область, куда будем копировать
        target_start_y = 0
        target_end_y = height - shift_down_pixels
        target_start_x = 0
        target_end_x = width - shift_left_pixels
        
        # 6. Выполняем смещение путем копирования данных
        if start_y < height and start_x < width:
            shifted_inc_ampl[target_start_y:target_end_y, target_start_x:target_end_x] = \
                inc_ampl[start_y:end_y, start_x:end_x]
        
        return shifted_inc_ampl
    
    else:
        print("Ошибка: пустые границы макропикселей")
        return inc_ampl

incident_mask = shift_inc_ampl_matrix(inc_ampl, x_bounds, y_bounds, shift_down=0, shift_left=0)
#-------------SLM intended influence---------------
def generate_amplitude_matrix(
    kind=None, 
    discretisation=False, 
    reference=False, 
    k=1, 
    x=None, 
    frequency = None, 
    same_size_as100 = False):

    am = torch.zeros(y_mac_size, x_mac_size)

    if kind == "ones":
        am = torch.ones(y_mac_size, x_mac_size)
    elif kind =="k_value":
        print(f"Chosen type of SLM: k_values\tk = {k}")
        am = torch.ones(y_mac_size, x_mac_size)*k
    elif kind == "chess":
        for i in range(y_mac_size):
            for j in range(x_mac_size):
                if (i+j) % 2 == 0:
                    am[i,j] = 1
    elif kind == "chess_rand":
        for i in range(y_mac_size):
            for j in range(x_mac_size):
                if (i+j) % 2 == 0:
                    am[i,j] = rnd.random()
    elif kind == "sin_y":
        for i in range(y_mac_size):
            
            am[i,:] = abs(np.sin(frequency*i))
    elif kind == "rand":
        am = x * torch.rand(y_mac_size, x_mac_size)

    elif kind == "x_rand":
        am = x+(1-x)*torch.rand(y_mac_size, x_mac_size)
    
    if same_size_as100 == True:
        am[9:, :] = 0    
        am[:, 15:] = 0 
    if reference == True:
        # Устанавливаем границу для reference строки
        reference_bound = 15 if same_size_as100 else x_mac_size
    
        for i in range(reference_bound):
            if i % 2 != 0: 
                am[1,i] = 1
    
    target = am

    if discretisation == True:
        target = torch.zeros(y_mac_size//2, x_mac_size//2)
        for i in range(y_mac_size):
            if i % 2 == 0: 
                am[i,:] = 0
        for j in range(x_mac_size):
            if j % 2 == 0: 
                am[:,j] = 0
        target = am[1::2, 1::2]
    
    return am, target, discretisation

t_ampl, matrix, matrix_discretisation = generate_amplitude_matrix("k_value", True, True, k=0.6, x = 1, same_size_as100 = True)
print(f"Matrix discretisation = {matrix_discretisation}")
print(f"Amplitude shape = {t_ampl.shape}")

def generate_slm_phase_mask(target_amplitudes=None, discretisation=None):
    """Генерация фазовой маски SLM для амплитудной модуляции"""
    
    # Проверка размеров
    assert x_res % mac_size == 0, "Высота SLM должна быть кратна размеру макропикселя"
    assert y_res % mac_size == 0, "Ширина SLM должна быть кратна размеру макропикселя"
    
    # Размеры матрицы макропикселей
    macro_height = y_res // mac_size
    macro_width = x_res // mac_size
    
    # Генерация случайных целевых амплитуд, если не предоставлены
    if target_amplitudes is None:
        target_amplitudes = torch.rand(macro_height, macro_width)
        print(f"Сгенерирована матрица целевых амплитуд размером {macro_height}x{macro_width}")
    
    # Создание пустой фазовой маски SLM
    phase_mask = torch.zeros(y_res, x_res)
    
    # Фаза θ для всех макропикселей
    theta = 0.0
    
    # Проход по всем макропикселям SLM
    for i in range(macro_height):
        for j in range(macro_width):
            
            # Текущая целевая амплитуда
            A_target = target_amplitudes[i, j]
            
            # Расчет фаз для основной области макропикселя
            if A_target > 0:
                psi = np.arccos(A_target)  # From A = cos(ψ) ==> ψ = arccos(A)
            else:
                psi = np.pi / 2  # Для A=0 используем π/2
            
            # Расчет фаз для двух групп пикселей основной области
            phi1 = theta + psi          # φ1 = θ + ψ
            phi2 = theta - psi          # φ2 = θ - ψ
            
            # Нормализация фаз к диапазону [0, 2π)
            phi1 = phi1 % (2 * np.pi)
            phi2 = phi2 % (2 * np.pi)
            
            phi1_zero = (theta + np.arccos(0.01))%(2 * np.pi)
            phi2_zero = (theta - np.arccos(0.01))%(2 * np.pi)
          
            # Определение координат текущего макропикселя в SLM
            start_row = i * mac_size
            start_col = j * mac_size
            
            # Определение границы обрезания
            if discretisation is not None:
                cut_boundary = mac_size - 2 * discretisation
            else:
                cut_boundary = mac_size
            
            # Заполнение макропикселя шахматным паттерном
            for row in range(mac_size):
                for col in range(mac_size):
                    
                    # ИСПРАВЛЕНИЕ: обрезаем только справа и снизу
                    in_cut_area = (discretisation is not None and 
                                 (row >= mac_size - discretisation or 
                                  col >= mac_size - discretisation))
                    
                    if in_cut_area:
                        # Обрезанная область - используем фазы для нулевой амплитуды
                        if (row + col) % 2 == 0 :           #and i != 1 - не применять на референсный ряд
                            phase_mask[start_row + row, start_col + col] = 1.571
                        else:
                            phase_mask[start_row + row, start_col + col] = 4.712
                    else:
                        # Основная область - используем фазы для целевой амплитуды
                        if (row + col) % 2 == 0:
                            phase_mask[start_row + row, start_col + col] = phi1
                        else:
                            phase_mask[start_row + row, start_col + col] = phi2
    
    return phase_mask
# Генерация фазовой маски SLM
mask = generate_slm_phase_mask(t_ampl, discretisation = 0)

# Обновление SLM с новой маской
slm = elements.SpatialLightModulator(
    simulation_parameters=params,
    mask=mask,
    height=slm_height,
    width=slm_width
)
slm_aperture = slm.get_aperture
slm_transmission_function = slm.transmission_function

#-------------First multiplication-----------------
incident_wave1 = w.mul(incident_wave, slm_aperture, ('H','W'), params)  # Apply aperture
incident_wave2 = w.mul(incident_wave1, inc_ampl, ('H','W'), params) # Apply vector
incident_wave3 = w.mul(incident_wave2, incident_mask, ('H','W'), params) # Apply mask to reduce macropixel size
first_intensity = incident_wave3.intensity
first_phase = incident_wave3.phase

#-------------Second multiplication----------------
# SLM уже содержит матричную модуляцию в своей фазовой маске
field_after_slm = slm.forward(incident_wavefront=incident_wave3)
phase_after_slm = field_after_slm.phase
intensity_after_slm = field_after_slm.intensity

#-------------Free space-----------------
f = 200 * ureg.mm
focal_space = elements.FreeSpace(
    simulation_parameters=params,
    distance=1*f,
    method="AS"
)
free_space = elements.FreeSpace(
    simulation_parameters=params,
    distance=1*f,
    method="AS"
)

#-------------Lens-----------------------
y_lens = CylindricalLens(
    simulation_parameters=params,
    focal_length=f,
    radius=30 * ureg.mm, 
    axis='y',  
)
lens = CylindricalLens(
    simulation_parameters=params,
    focal_length=2*f,
    radius=100 * ureg.mm, 
    axis='x',  
)
s_lens = elements.ThinLens(
    simulation_parameters=params,
    focal_length=f,
    radius=100 * ureg.mm
)
#------------General Setup---------------
setup = LinearOpticalSetup([focal_space, y_lens, focal_space, lens, focal_space, y_lens, focal_space])
field_after_lens = setup.forward(input_wavefront=field_after_slm)
intensity_after_lens = field_after_lens.intensity
phase_after_lens = field_after_lens.phase

#-----------For Representation--------------
setup1 = LinearOpticalSetup([focal_space])
field_after_lens11 = setup1.forward(input_wavefront=field_after_slm)
intensity_after_slm_end = field_after_lens11.intensity

#-----------Calculations: Theory------------
def extract_target_value(tv, tm):
    result = torch.zeros(tm.shape[0])
    for i in range(tm.shape[0]):
        value = 0
        for j in range(tm.shape[1]):
            value += tm[i,j] * tv[j]
        result[i] = value
    result = result / result.max()
    return result

theory_result = extract_target_value(target_v, matrix)
print("Theory result = ", theory_result)

#-----------Calculations: Experiment-------
def extract_actual_value(center, x_range, tm, light):
    result = torch.zeros(tm.shape[0])
    
    # Берем полосу вокруг центра
    intensity = light[y_min:y_max, center - x_range: center + x_range]
    
    # Вариант без дискретизации не проверялся, мб там есть ошибки.
    if matrix_discretisation == False:
        for i in range(y_mac_size):
            start_y, end_y = y_bounds[i]
            # Используем точные границы макропикселя
            target = intensity[start_y-y_min:end_y-y_min, :].sum() * (ly/Ny) * (lx/Nx)
            result[i] = torch.sqrt(target)  

    if matrix_discretisation == True:
        for i in range(0, y_mac_size, 2):
            start_y, end_y = y_bounds[i]
            # Используем точные границы макропикселя
            target = intensity[start_y-y_min:end_y-y_min, :].sum() * (ly/Ny) * (lx/Nx)
            result[i//2] = torch.sqrt(target)  
    
    result1 = torch.flip(result, dims=[0])
    norm = result1[0]
    if norm > 0:
        result1 = torch.div(result1, norm)
    
    return result1

actual_result = extract_actual_value(Nx//2, 10, matrix, intensity_after_lens)
print("Actual result = ", actual_result)

#-----------Error calculation--------------
def error_calc(tv, av):
    abs_error = torch.zeros(len(tv))
    rel_error = torch.zeros(len(tv))
    for i in range(len(tv)):
        abs_error[i] = abs(av[i]-tv[i])
        if tv[i] != 0:
            rel_error[i] = abs(av[i]-tv[i])/tv[i] * 100
        else:
            rel_error[i] = torch.tensor(float('nan'))
    return abs_error, rel_error

a_err, r_err = error_calc(theory_result, actual_result)
print("Absolute error: ", a_err)
print(f"Relative error (%): ", r_err)

#-----------Plots--------------------------
'''
fig, ax = plt.subplots(
    1, 2, figsize=(15, 10), edgecolor='black', linewidth=3, frameon=True
)

im1 = ax[0].pcolormesh(x_grid, y_grid, first_phase, cmap='inferno')
ax[0].set_aspect('equal')
ax[0].set_title("Beam's phase distribution \n before propagating \n through the SLM")
ax[0].set_xlabel('$x$ [m]')
ax[0].set_ylabel('$y$ [m]')
fig.colorbar(im1, ax=ax[0])

im2 = ax[1].pcolormesh(x_grid, y_grid, phase_after_slm, cmap='inferno')
ax[1].set_aspect('equal')
ax[1].set_title("Beam's phase distribution \n after propagating \n through the SLM")
ax[1].set_xlabel('$x$ [m]')
ax[1].set_ylabel('$y$ [m]')
fig.colorbar(im2, ax=ax[1])

plt.show()

'''
'''
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

im1 = ax[0].pcolormesh(x_grid, y_grid, first_intensity, cmap='inferno')
ax[0].set_aspect('equal')
ax[0].set_title("After applying vector")
ax[0].set_xlabel('$x$ [m]')
ax[0].set_ylabel('$y$ [m]')
fig.colorbar(im1, ax=ax[0])

im2 = ax[1].pcolormesh(x_grid, y_grid, intensity_after_slm_end, cmap='inferno')
ax[1].set_aspect('equal')
ax[1].set_title("After SLM (matrix modulation)")
ax[1].set_xlabel('$x$ [m]')
ax[1].set_ylabel('$y$ [m]')
fig.colorbar(im2, ax=ax[1])

im3 = ax[2].pcolormesh(x_grid, y_grid, intensity_after_lens, cmap='inferno')
ax[2].set_aspect('equal')
ax[2].set_title("After lens system")
ax[2].set_xlabel('$x$ [m]')
ax[2].set_ylabel('$y$ [m]')
fig.colorbar(im3, ax=ax[2])

plt.show()
'''
#----------Files initialization-------------

def initialize_error_file(filename=f"matrix_errors_analysis_mac{mac_size}_Nx{Nx}.csv"):
    """Инициализирует файл с заголовками"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("k_value; avg_abs_err; max_abs_err; avg_rel_error; max_rel_error\n")

def append_error_to_file(k_value, a_err, r_err, filename="error_results.csv"):
    """Добавляет строку с результатами в файл"""
    
    # Конвертируем тензоры в списки
    a_err_list = a_err.detach().numpy().tolist()
    r_err_list = r_err.detach().numpy().tolist()
    avg_abs_error = np.mean(a_err_list)
    max_abs_error = np.max(a_err_list)
    avg_rel_error = np.nanmean(r_err_list)  # Игнорируем NaN значения
    max_rel_error = np.nanmax(r_err_list)


    
    # Записываем строку в файл
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{k_value};{avg_abs_error};{max_abs_error};{avg_rel_error};{max_rel_error}\n")

def initialize_vector_file(filename=f"matrix_errors_analysis_mac{mac_size}_Nx{Nx}.csv"):
    """Инициализирует файл с заголовками"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Value; target;actual; abs_err; rel_error \n")

def append_vectors_to_file(target, actual, abs_err, rel_err, filename, size100):
    l = len(target)
    if size100 == True:
        l=3
    with open(filename, 'a', encoding='utf-8') as f:
        for i in range(1,l,1):    # ЗАМЕНИТЬ 3 на len(target)
            a=target[i].item()
            b=actual[i].item()
            c=abs_err[i].item()
            d=rel_err[i].item()
            f.write(f"{a};{a};{b};{c};{d}\n")

def count_values_less_than_01(vector, matrix):
    if isinstance(vector, torch.Tensor):
        vector_np = vector.detach().cpu().numpy()
    else:
        vector_np = np.array(vector)
        
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = np.array(matrix)
    
    # Подсчет для вектора
    vector_count = np.sum(vector_np < 0.1)
    vector_total = vector_np.size
    vector_percentage = (vector_count / vector_total) * 100
    
    # Подсчет для матрицы
    matrix_count = np.sum(matrix_np < 0.1)
    matrix_total = matrix_np.size
    matrix_percentage = (matrix_count / matrix_total) * 100
    return vector_count, matrix_count
#----------Lots of error calculation--------

def error_data_whole(same_size = False):
    initialize_error_file(f"matrix_errors_analysis_mac{mac_size}_Nx{Nx}.csv")
    
    for j in np.linspace(0.01,1,50):
        kk=j.round(2)
        print(f"=== Iteration {round(j*100)}/100, k = {kk} ===")
        mask = torch.zeros(y_res, x_res)
        gc.collect()
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        slm = elements.SpatialLightModulator(
    simulation_parameters=params,
    mask=mask,
    height=slm_height,
    width=slm_width
)
        x_bounds, y_bounds, x_min, x_max, y_min, y_max = calculate_macro_coordinates()

        incident_wave = w.Wavefront.plane_wave(
   simulation_parameters=params,
   distance=1 * ureg.cm,
   wave_direction=[0, 0, 1]
)

        vect, target_v = vector_gen("ones", True, same_as100 = same_size)
        inc_ampl = incident_amplitude(vect, True, True)

        t_ampl, matrix, matrix_discretisation = generate_amplitude_matrix("k_value", True, True, kk, same_size_as100 = same_size)

        mask = generate_slm_phase_mask(t_ampl)
        slm = elements.SpatialLightModulator(
    simulation_parameters=params,
    mask=mask,
    height=slm_height,
    width=slm_width
)
        slm_aperture = slm.get_aperture
        
        incident_wave1 = w.mul(incident_wave, slm_aperture, ('H','W'), params)  # Apply aperture
        incident_wave2 = w.mul(incident_wave1, inc_ampl, ('H','W'), params) # Apply vector
        
        setup = LinearOpticalSetup([slm, focal_space, y_lens, focal_space, lens, focal_space, y_lens, focal_space])
        field_after_lens = setup.forward(input_wavefront=incident_wave2)
        intensity_after_lens = field_after_lens.intensity
        theory_result = extract_target_value(target_v, matrix)
        actual_result = extract_actual_value(Nx//2, 4, matrix, intensity_after_lens)
        a_err, r_err = error_calc(theory_result, actual_result)
        
        append_error_to_file(j, a_err, r_err, f"matrix_errors_analysis_mac{mac_size}_Nx{Nx}.csv")
    return 0



def error_data_macro_size():
    
    for discr_value in range(10, 90, 10):  # Изменил имя переменной
        filename = f"matrix_errors_analysis_size_reduction{discr_value}_Nx{Nx}.csv"
        
        initialize_error_file(filename)
        print(f"\n\n===Calculations for pixel of size {mac_size-2*discr_value}x{mac_size-2*discr_value}")
        
        # Запись размера пикселя в файл
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"pixel size = {mac_size-2*discr_value}\n")  # Исправлена формула

        for j_idx, j in enumerate(np.linspace(0.01, 1, 20)):
            kk = round(float(j), 2)
            print(f"=== Iteration {j_idx+1}/20, k = {kk} ===")
            
            # Очистка памяти
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            mask = torch.zeros(y_res, x_res)
            print(f"Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'CPU mode'}")
            
            slm = elements.SpatialLightModulator(
                simulation_parameters=params,
                mask=mask,
                height=slm_height,
                width=slm_width
            )
            
            x_bounds, y_bounds, x_min, x_max, y_min, y_max = calculate_macro_coordinates()

            incident_wave = w.Wavefront.plane_wave(
                simulation_parameters=params,
                distance=1 * ureg.cm,
                wave_direction=[0, 0, 1]
            )

            vect, target_v = vector_gen("ones", True)  # 1) Vector
            inc_ampl = incident_amplitude(vect, True, True)  # 2) Vector amplitude
            
            # 3) Incident vector macropixel size reduction - ДОБАВЛЕНО
            incident_mask = shift_inc_ampl_matrix(inc_ampl, x_bounds, y_bounds, 
                                                shift_down=0, shift_left=0)
            
            t_ampl, matrix, matrix_discretisation = generate_amplitude_matrix("k_value", True, True, kk)  # 4) Matrix

            mask = generate_slm_phase_mask(t_ampl, discretisation=0)  # 5) SLM Amplitude modulation matrix
            
            slm = elements.SpatialLightModulator(
                simulation_parameters=params,
                mask=mask,
                height=slm_height,
                width=slm_width
            )
            slm_aperture = slm.get_aperture
        
            incident_wave1 = w.mul(incident_wave, slm_aperture, ('H','W'), params)  # Apply aperture
            incident_wave2 = w.mul(incident_wave1, inc_ampl, ('H','W'), params)  # Apply vector
            incident_wave3 = w.mul(incident_wave2, incident_mask, ('H','W'), params)  # Apply reduction mask

            setup = LinearOpticalSetup([slm, focal_space, y_lens, focal_space, lens, focal_space, y_lens, focal_space])
            field_after_lens = setup.forward(input_wavefront=incident_wave3)
            intensity_after_lens = field_after_lens.intensity
            theory_result = extract_target_value(target_v, matrix)
            actual_result = extract_actual_value(Nx//2, 4, matrix, intensity_after_lens)
            a_err, r_err = error_calc(theory_result, actual_result)
        
            append_error_to_file(kk, a_err, r_err, filename)  # Исправлено: kk вместо j
    return 0


def rand_vector_compare(same_size):
    file = f"vector_comparison_size{mac_size}_Nx{Nx}.csv"
    #initialize_vector_file(file)

    for k in range(16):
        print(f"====== Calculation for {mac_size}x{mac_size}. Instance {k+1} out of 16 ======\n")
        
        
        if k<=8:
            s = (k+1)*0.1
            rand_type="rand"
            print(f"-Chosen type = {rand_type}\n Range of values = 0:{s}")
        elif k>8 and k<14:
            s = (k-6)*0.1
            rand_type="x_rand"
            print(f"-Chosen type = {rand_type}\n Range of values = {s}:1")
        elif k>=14:
            rand_type = "rand"
            print(f"-Chosen type = {rand_type}\n Range of values = 0:1")
            s = 1
        
        
        # Очистка памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        mask = torch.zeros(y_res, x_res)
        print(f"Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'CPU mode'}")
            
        slm = elements.SpatialLightModulator(
            simulation_parameters=params,
            mask=mask,
            height=slm_height,
            width=slm_width
        )
            
        x_bounds, y_bounds, x_min, x_max, y_min, y_max = calculate_macro_coordinates()

        incident_wave = w.Wavefront.plane_wave(
            simulation_parameters=params,
            distance=1 * ureg.cm,
            wave_direction=[0, 0, 1]
        )

        vect, target_v = vector_gen(rand_type, True, x = s, same_as100 = same_size)  # 1) Vector
        inc_ampl = incident_amplitude(vect, True, True)  # 2) Vector amplitude
        t_ampl, matrix, matrix_discretisation = generate_amplitude_matrix(rand_type, True, True , x = s, same_size_as100 = same_size)  # 4) Matrix
        mask = generate_slm_phase_mask(t_ampl)  # 5) SLM Amplitude modulation matrix
            
        slm = elements.SpatialLightModulator(
            simulation_parameters=params,
            mask=mask,
            height=slm_height,
            width=slm_width
        )
        slm_aperture = slm.get_aperture
        
        incident_wave1 = w.mul(incident_wave, slm_aperture, ('H','W'), params)  # Apply aperture
        incident_wave2 = w.mul(incident_wave1, inc_ampl, ('H','W'), params)  # Apply vector
    

        setup = LinearOpticalSetup([slm, focal_space, y_lens, focal_space, lens, focal_space, y_lens, focal_space])
        field_after_lens = setup.forward(input_wavefront=incident_wave2)
        intensity_after_lens = field_after_lens.intensity
        theory_result = extract_target_value(target_v, matrix)
        actual_result = extract_actual_value(Nx//2, 10, matrix, intensity_after_lens)
        print(theory_result)
        print(actual_result)
        a_err, r_err = error_calc(theory_result, actual_result)

        append_vectors_to_file(theory_result, actual_result, a_err, r_err, file, size100 = same_size)


