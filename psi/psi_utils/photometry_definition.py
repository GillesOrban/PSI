# Proposed convention: 
#      <instrument_name>_<band>
PHOT = {
       'METIS_L':  {'lam': 3.81e-6,
              'pscale': 5.47,                         # [mas]
              'flux_star': 8.999e+10,                 # HCI-L long
              'flux_bckg': 8.878e+04},
       'METIS_M':  {'lam': 4.79e-6,
              'pscale': 5.47,
              'flux_star': 2.452e+10,                 # CO ref
              'flux_bckg': 6.714e+05},
       'METIS_N1': {'lam': 8.70e-6,
              'pscale': 6.79,
              'flux_star': 3.684e+10,                 # GeoSnap N1
              'flux_bckg': 4.725e+07},
       'METIS_N2': {'lam': 11.33e-6,
              'pscale': 6.79,
              'flux_star': 3.695e+10,                 # GeoSnap N2
              'flux_bckg': 1.122e+08},
       'METIS_N1a': {'lam': 8.67e-6,
              # 'pscale': 10.78,
              'flux_star': 2.979e+10,                 # Aquarius N1
              'flux_bckg': 9.630e+07},
       'N2a': {'lam': 11.21e-6,
              # 'pscale': 10.78,
              'flux_star': 2.823e+10,                 # Aquarius N2
              'flux_bckg': 2.142e+08}
       }
