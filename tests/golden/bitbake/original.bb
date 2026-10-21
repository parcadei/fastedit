SUMMARY = "Sample recipe"
LICENSE = "MIT"

do_install() {
    install -d ${D}${bindir}
}
