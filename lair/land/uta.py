"""
Get UTA vehicle data.

    UTA is dedicated to providing free real-time developer API's to the UTA transit system.
    Instead of developing our own standard we chose to implement the SIRI standard.
"""

import os


class SIRI:
    """
    UTA SIRI API.

    SIRI (Service Interface for Real Time Information) is an XML standard
    covering a wide range of types of real-time information for public
    transportation. This standard is has been adopted by the European
    standards-setting body CEN. This standard is not owned by any one vendor
    or public transportation agency or operator. It has been adopted by a
    growing number of different projects by vendors and agencies around
    North America and Europe.

    SIRI Schema: https://github.com/SIRI-CEN/SIRI/tree/master
    Resources: https://data4pt-project.eu/wp-content/uploads/2021/06/7-SIRI-Resources-and-documentation.pdf
    """

    api: str = 'http://api.rideuta.com/SIRI/SIRI.svc'

    def __init__(self, token=None):
        self.token = token or os.getenv('UTA_API_KEY')


class VehicleMonitoring(SIRI):
    """
    Vehicle monitoring API.

    This API provides real-time vehicle monitoring data for the UTA transit
    system. This API is part of the SIRI standard.
    """

    api: str = f'{SIRI.api}/VehicleMonitor'

    def __init__(self, token=None):
        super().__init__(token)

    def get_vehicle(self, vehicle_id: int | str, onwardcalls: bool = False) -> dict:
        """
        Get vehicle data.

        Returns
        -------
        dict
            The vehicle data.
        """
        pass

    def by_route(self, route: int | str, onwardcalls: bool = False) -> dict:
        """
        Get vehicle data by route.

        Returns
        -------
        dict
            The vehicle data.
        """
        pass
