pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T
import ".."

/*!
    TranslatePage (docs/GLASS_DESIGN.md §2.3): card "Languages" (Source with
    Auto-detect first, Target), card "Series" (the quick series-name context
    for the Gemini provider) and card "Session" (Start/Stop primary toggle,
    Grab mode, Show / hide glass) with the current hotkeys as hints.
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
    Accessible.name: "Translate"

    Column {
        id: column
        width: page.width
        spacing: Theme.gap

        GlassCard {
            title: "Languages"
            FormRow {
                label: "Source"
                GlassComboBox {
                    objectName: "sourceCombo"
                    model: bridge.languagesSource
                    value: bridge.sourceLang
                    onSelected: function (v) { bridge.sourceLang = v }
                }
            }
            FormRow {
                label: "Target"
                GlassComboBox {
                    objectName: "targetCombo"
                    model: bridge.languagesTarget
                    value: bridge.targetLang
                    onSelected: function (v) { bridge.targetLang = v }
                }
            }
        }

        GlassCard {
            title: "Series"
            FormRow {
                label: "Series name"
                hint: bridge.backendGemini
                      ? "Fills [Series Name] in the Gemini system prompt (Engines → Series context), e.g. Jujutsu Kaisen."
                      : "Used by the Gemini provider only (Engines → Translation → gemini)."
                GlassTextField {
                    id: seriesField
                    objectName: "seriesNameField"
                    placeholder: "e.g. Jujutsu Kaisen"
                    Binding on text { value: bridge.seriesName }
                    onCommitted: function (t) { bridge.seriesName = t; seriesField.text = bridge.seriesName }
                }
            }
        }

        GlassCard {
            title: "Session"
            FormRow {
                label: "Pipeline"
                hint: "Hotkey " + bridge.hotkeyRunning
                GlassButton {
                    id: startButton
                    objectName: "startButton"
                    primary: true
                    checkable: true
                    text: bridge.running ? "Stop" : "Start"
                    Binding on checked { value: bridge.running }
                    onClicked: {
                        bridge.startStop(!bridge.running)
                        startButton.checked = bridge.running
                    }
                }
            }
            FormRow {
                label: "Grab mode"
                hint: "Hotkey " + bridge.hotkeyGrab
                stretch: false
                GlassButton {
                    objectName: "grabButton"
                    text: "Grab mode"
                    onClicked: bridge.grabMode()
                }
            }
            FormRow {
                label: "Glass"
                hint: "Hotkey " + bridge.hotkeyHidden
                stretch: false
                GlassButton {
                    objectName: "glassButton"
                    text: "Show / hide glass"
                    onClicked: bridge.toggleGlass()
                }
            }
        }
    }
}
