# Per-generator CPack settings (CPACK_PROJECT_CONFIG_FILE). Linux trees stage through DESTDIR with the configured
# install prefix (/usr), so an archive must not add the prefix again (it would look for usr/usr/bin).
if(CPACK_SET_DESTDIR AND CPACK_GENERATOR MATCHES "^(TGZ|TXZ|TBZ2|ZIP)$")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/")
endif()
