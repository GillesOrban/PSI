from psi.psiSensor import PsiSensor


# config_file = 'config/config_demo_metis_compass.py'
config_file = 'config/config_metis_compass.py'
# config_file = 'config/config_hcipysim.py'


psi_sensor = PsiSensor(config_file)

psi_sensor.setup()
# Test: doing one iteration
psi_sensor.logger.info('Inputs:')
psi_sensor.evaluateSensorEstimate()
# psi_sensor.next()
# psi_sensor.evaluateSensorEstimate()

psi_sensor.loop(display=False)
