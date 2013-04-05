#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from numpy import zeros, mat, matrix, sqrt, sin, arange, pi, exp, cos, array, dot, max, abs, int16, eye

# Anzahl Filter
filters = 30;
m = arange(1., filters+1.);

# Saiten-Koeffizienten
l = 0.65;
Ts = 60.97;
rho = 1140;
A = 0.5188*10**-6;
E = 5.4*10**9;
I = 0.171*10**-12;
d1 = 8*10**-5;
d3 = -1.4*10**-5;

# Abtastpunkt
xa = 0.1;

# Abtastrate und Samplelänge
T = 44100;
seconds = 10;
samples = seconds*T;
y = zeros((1,samples))

# Blockverarbeitungs-Länge
blocksize = 100;

# Ausgangssignal
y = zeros((1,samples));

# Ausgangs- und Übergangsmatrix, Zustandsvektor
block_C = zeros((1, filters*2));
block_A = zeros((2*filters, 2*filters));
block_Apow = zeros(block_A.shape);
block_CA = zeros((blocksize, 2*filters));
block_state = zeros((2*filters, 1));

sigmas = zeros((1, filters));
omegas = zeros((1, filters));

for i in m:
	# Pol aufstellen
	gamma = i*(pi/l);
	sigma = (1/(2*rho*A)) * (d3*gamma**2 - d1);
	omega = sqrt( ( (E*I)/(rho*A) - (d3**2)/((2*rho*A)**2) )* gamma**4 + (Ts/(rho*A)+(d1*d3)/2*(rho*A)**2) * gamma**2 + (d1/(2*rho*A))**2);

	# Ausgangsgewichtung
	a = sin(i*pi*xa/l);

	# Übertrangungsfunktions-Koeffizienten
	b = T*sin(omega*1/T)/(omega*1/T);
	c1 = -2*exp(sigma*1/T)*cos(omega*1/T);
	c0 = exp(2*sigma*1/T);
		
	# Zustandsraum-Matrizen
	fA = array([[0, -c0], [1, -c1]]);
	fC = array([[0, a]]);
	state = array([[1, 0]]);

	# 1-Zeilen Ausgangsmatrix
	block_C[0,(i-1)*2:(i-1)*2+2] = fC
	omegas[0,i-1] = omega
	sigmas[0,i-1] = sigma
	block_A[(i-1)*2:(i-1)*2+2,(i-1)*2:(i-1)*2+2] = fA
	block_state[(i-1)*2:(i-1)*2+2,0] = state;

block_Apow = eye(block_A.shape[1])

j = 0;
while j < blocksize:
	block_Apow = dot(block_Apow,block_A);
	block_CA[j] = dot(block_C,block_Apow);
	j = j + 1;

j = 0;
while j < samples:
	y[0:1, j:j+blocksize] = dot(block_CA, block_state).T;
	block_state = dot(block_Apow, block_state);
	j = j + blocksize;

y = np.int16(y.T/np.max(np.abs(y)) * 32767)
write('out.wav', 44100, y)