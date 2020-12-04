#!/usr/bin/env python3

def OracleDBConnector():
    import cx_Oracle

    username = 'crystal'
    password = 'crystal'
    dsn = 'clonedb429:1521/CRYSTAL.RIGAKU.COM'

    connection = cx_Oracle.connect(username,
                                password,
                                dsn,
                                encoding = 'utf-8')


OracleDBConnector()