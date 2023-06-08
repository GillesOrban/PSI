


asym_telDiam=40
asym_nsteps = 20
vac_coords= kernel.kpo.kpi.VAC

step_size = asym_telDiam / asym_nsteps

nbs=asym_nsteps
phase = np.zeros((nbs-1, nbs-1))

coords = np.array(vac_coords[:,0:2] / step_size + (asym_telDiam/2) / step_size, dtype='int')
phase[list(coords[:,1]-1), list(coords[:,0]-1)] = np.append(0, kernel._wft)#vac_coords[:,2]


plt.figure()
plt.imshow(phase, vmin=-0.5, vmax=0.5)
