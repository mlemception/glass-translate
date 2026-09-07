pragma ComponentBehavior: Bound
import QtQuick

/*!
    Glyph - a small monochrome icon from the Windows icon font
    ("Segoe Fluent Icons", falling back to "Segoe MDL2 Assets", which shares the
    code points).  `name` is one of the keys below; any other string is drawn
    verbatim.  Colour defaults to Theme.ink, size to 10 px scaled by the
    Windows text size.
*/
Text {
    id: glyph

    /*! close | minimize | chevron-down | chevron-up | check | plus | minus | folder */
    property string name: ""
    property real size: Math.round(10 * Theme.textScale)

    readonly property var codes: ({
        "close": "\uE8BB",
        "minimize": "\uE921",
        "chevron-down": "\uE70D",
        "chevron-up": "\uE70E",
        "check": "\uE73E",
        "plus": "\uE710",
        "minus": "\uE738",
        "folder": "\uE8B7"
    })
    readonly property string iconFamily: Qt.fontFamilies().indexOf("Segoe Fluent Icons") >= 0
                                         ? "Segoe Fluent Icons" : "Segoe MDL2 Assets"

    text: codes[name] !== undefined ? codes[name] : name
    font.family: iconFamily
    font.pixelSize: size
    color: Theme.ink
    horizontalAlignment: Text.AlignHCenter
    verticalAlignment: Text.AlignVCenter
    renderType: Text.NativeRendering
}
