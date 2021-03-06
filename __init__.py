# -*- coding: utf-8 -*-
"""
/***************************************************************************
 preprocessing_optimization
                                 A QGIS plugin
 preprocessing_optimization
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2021-04-10
        copyright            : (C) 2021 by Moritz
        email                : Moritz
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
    """Load preprocessing_optimization class from file preprocessing_optimization.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .preprocessing_optimization import preprocessing_optimization
    return preprocessing_optimization(iface)
