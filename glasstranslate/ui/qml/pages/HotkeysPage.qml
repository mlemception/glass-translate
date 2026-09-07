pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T
import ".."

/*!
    HotkeysPage (docs/GLASS_DESIGN.md §2.3, §3): card "Global hotkeys" with the
    three monospace fields.  A commit calls bridge.setHotkey(which, text): on
    success the field shows the normalized form; empty text reverts silently;
    invalid text reverts, shakes and leaves the message in the status strip.
*/
// qmllint disable unqualified
Flickable {
    id: page

    contentWidth: width
    contentHeight: column.implicitHeight
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    flickableDirection: Flickable.VerticalFlick
    T.ScrollBar.vertical: GlassScrollBar {}

    Accessible.role: Accessible.Pane
    Accessible.name: "Hotkeys"

    function commitHotkey(field, which, text) {
        var ok = bridge.setHotkey(which, text)
        field.text = which === "grab" ? bridge.hotkeyGrab
                   : which === "running" ? bridge.hotkeyRunning : bridge.hotkeyHidden
        if (!ok && text.trim().length > 0)
            field.error = true
    }

    Column {
        id: column
        width: page.width
        spacing: Theme.gap

        GlassCard {
            title: "Global hotkeys"
            Text {
                width: parent.width
                text: "pynput syntax, e.g. <ctrl>+<alt>+g"
                font.family: Theme.family
                font.pointSize: Theme.fontCaption
                color: Theme.inkSecondary
                wrapMode: Text.WordWrap
            }
            FormRow {
                label: "Toggle grab mode"
                GlassTextField {
                    id: grabField
                    objectName: "hotkeyGrabField"
                    monospace: true
                    Binding on text { value: bridge.hotkeyGrab }
                    onCommitted: function (t) { page.commitHotkey(grabField, "grab", t) }
                }
            }
            FormRow {
                label: "Start / stop"
                GlassTextField {
                    id: runningField
                    objectName: "hotkeyRunningField"
                    monospace: true
                    Binding on text { value: bridge.hotkeyRunning }
                    onCommitted: function (t) { page.commitHotkey(runningField, "running", t) }
                }
            }
            FormRow {
                label: "Show / hide glass"
                GlassTextField {
                    id: hiddenField
                    objectName: "hotkeyHiddenField"
                    monospace: true
                    Binding on text { value: bridge.hotkeyHidden }
                    onCommitted: function (t) { page.commitHotkey(hiddenField, "hidden", t) }
                }
            }
        }
    }
}
