pragma ComponentBehavior: Bound
import QtQuick

/*!
    GlassFontComboBox - GlassComboBox over a plain list of family names
    (bridge.fontFamilies, 300+ items): both roles empty, the list reuses its
    delegates, type-ahead comes from T.ComboBox, and every item (plus the
    closed state) is drawn in the family it names.
*/
GlassComboBox {
    textRole: ""
    valueRole: ""
    fontPreview: true
    label: "Font"          // accessible name when no FormRow supplies one
}
