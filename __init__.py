# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SHARAD3d
                                 A QGIS plugin
 SHARAD3d
                             -------------------
        begin                : 2018-02-01
        copyright            : (C) 2018 by SITING XIONG
        email                : siting.xiong.14@ucl.ac.uk
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load SHARAD3D class from file SHARAD3D.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .SHARAD3d import SHARAD3d
    return SHARAD3d(iface)
