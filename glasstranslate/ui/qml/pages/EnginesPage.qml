pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T
import ".."

/*!
    EnginesPage (docs/GLASS_DESIGN.md §2.3, §3): card "OCR" (engine, device);
    card "Translation" (backend, translate device - enabled iff argos, API URL /
    API key - enabled iff libretranslate, Models dir + Browse always enabled,
    Download model / Get Sugoi - enabled iff argos and no download running, the
    inline progress row with Hide); card "Pipeline" (refresh rate, debounce,
    min OCR confidence steppers).
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
    Accessible.name: "Engines"

    Column {
        id: column
        width: page.width
        spacing: Theme.gap

        GlassCard {
            title: "OCR"
            FormRow {
                label: "OCR engine"
                GlassComboBox {
                    objectName: "ocrEngineCombo"
                    model: bridge.ocrEngines
                    value: bridge.ocrEngine
                    onSelected: function (v) { bridge.ocrEngine = v }
                }
            }
            FormRow {
                label: "OCR device"
                GlassComboBox {
                    objectName: "ocrDeviceCombo"
                    model: bridge.ocrDevices
                    value: bridge.ocrDevice
                    onSelected: function (v) { bridge.ocrDevice = v }
                }
            }
        }

        GlassCard {
            title: "Translation"
            FormRow {
                label: "Translation"
                GlassComboBox {
                    objectName: "backendCombo"
                    model: bridge.backends
                    value: bridge.translationBackend
                    onSelected: function (v) { bridge.translationBackend = v }
                }
            }
            FormRow {
                label: "Translate device"
                hint: bridge.translateDeviceHint
                GlassComboBox {
                    objectName: "translateDeviceCombo"
                    enabled: bridge.backendArgos
                    model: bridge.translateDevices
                    value: bridge.translateDevice
                    onSelected: function (v) { bridge.translateDevice = v }
                }
            }
            FormRow {
                label: "API URL"
                GlassTextField {
                    id: apiUrlField
                    objectName: "apiUrlField"
                    enabled: bridge.backendOnline
                    placeholder: "https://…"
                    Binding on text { value: bridge.apiUrl }
                    // the bridge strips / rejects some input: always show what was stored
                    onCommitted: function (t) { bridge.apiUrl = t; apiUrlField.text = bridge.apiUrl }
                }
            }
            FormRow {
                label: "API key"
                GlassTextField {
                    id: apiKeyField
                    objectName: "apiKeyField"
                    enabled: bridge.backendOnline
                    password: true
                    Binding on text { value: bridge.apiKey }
                    onCommitted: function (t) { bridge.apiKey = t; apiKeyField.text = bridge.apiKey }
                }
            }
            FormRow {
                label: "Models dir"
                Item {
                    implicitHeight: modelsDirField.implicitHeight
                    GlassTextField {
                        id: modelsDirField
                        objectName: "modelsDirField"
                        anchors.left: parent.left
                        anchors.right: browseButton.left
                        anchors.rightMargin: Theme.gap
                        Binding on text { value: bridge.modelsDir }
                        // an empty / blank dir keeps the previous value (section 3): show it again
                        onCommitted: function (t) { bridge.modelsDir = t; modelsDirField.text = bridge.modelsDir }
                    }
                    GlassButton {
                        id: browseButton
                        objectName: "browseButton"
                        anchors.right: parent.right
                        text: "Browse…"
                        onClicked: bridge.browseModelsDir()
                    }
                }
            }
            FormRow {
                label: "Models"
                stretch: false
                Row {
                    spacing: Theme.gap
                    GlassButton {
                        objectName: "downloadButton"
                        text: "Download model…"
                        enabled: bridge.backendArgos && !bridge.downloadActive
                        onClicked: bridge.downloadModel(false)
                    }
                    GlassButton {
                        objectName: "sugoiButton"
                        text: "Get Sugoi (ja→en)…"
                        hint: "Install the Sugoi v4 Japanese→English model (~700 MB). Trained on game and visual-novel dialogue, it handles short manga lines much better than the generic Argos ja→en package and takes priority for that pair."
                        enabled: bridge.backendArgos && !bridge.downloadActive
                        onClicked: bridge.downloadModel(true)
                    }
                }
            }
            FormRow {
                objectName: "downloadRow"
                visible: bridge.downloadVisible
                label: "Download"
                Item {
                    id: progressRow
                    implicitHeight: hideButton.implicitHeight + progressBar.height + 6
                    Text {
                        id: progressLabel
                        anchors.left: parent.left
                        anchors.right: hideButton.left
                        anchors.rightMargin: Theme.gap
                        anchors.verticalCenter: hideButton.verticalCenter
                        text: bridge.downloadLabel
                        elide: Text.ElideMiddle
                        font.family: Theme.family
                        font.pointSize: Theme.fontBody
                        color: Theme.ink
                    }
                    GlassButton {
                        id: hideButton
                        objectName: "hideDownloadButton"
                        anchors.right: parent.right
                        anchors.top: parent.top
                        text: "Hide"
                        onClicked: bridge.hideDownload()
                    }
                    Item {
                        id: progressBar
                        anchors.left: parent.left
                        anchors.right: hideButton.left
                        anchors.rightMargin: Theme.gap
                        anchors.top: hideButton.bottom
                        anchors.topMargin: 6
                        height: 6
                        readonly property bool indeterminate: bridge.downloadProgress < 0
                        Accessible.role: Accessible.ProgressBar
                        Accessible.name: bridge.downloadLabel
                        Rectangle {
                            anchors.fill: parent
                            radius: 3
                            color: Theme.mode === "hc" ? Theme.sysPal.windowText : Theme.fillTrack
                        }
                        Rectangle {
                            id: progressFill
                            height: parent.height
                            radius: 3
                            color: Theme.accent
                            width: progressBar.indeterminate ? parent.width * 0.35
                                                             : parent.width * Math.min(100, bridge.downloadProgress) / 100
                            x: progressBar.indeterminate ? progressBar.sweep * (progressBar.width - width) : 0
                        }
                        // the only loop in the UI: runs while an indeterminate download row is visible
                        property real sweep: 0
                        SequentialAnimation on sweep {
                            running: progressBar.visible && progressBar.indeterminate && !Theme.reduceMotion
                            loops: Animation.Infinite
                            NumberAnimation { to: 1; duration: 900; easing.type: Easing.InOutSine }
                            NumberAnimation { to: 0; duration: 900; easing.type: Easing.InOutSine }
                        }
                    }
                }
            }
        }

        GlassCard {
            title: "Pipeline"
            FormRow {
                label: "Refresh rate"
                GlassStepper {
                    objectName: "refreshStepper"
                    minimum: 0.5
                    maximum: 60
                    step: 0.5
                    decimals: 1
                    suffix: " Hz"
                    number: bridge.refreshHz
                    onCommitted: function (v) { bridge.refreshHz = v }
                }
            }
            FormRow {
                label: "Debounce"
                GlassStepper {
                    objectName: "debounceStepper"
                    minimum: 0
                    maximum: 2000
                    step: 10
                    decimals: 0
                    suffix: " ms"
                    number: bridge.debounceMs
                    onCommitted: function (v) { bridge.debounceMs = v }
                }
            }
            FormRow {
                label: "Min OCR confidence"
                GlassStepper {
                    objectName: "confidenceStepper"
                    minimum: 0
                    maximum: 1
                    step: 0.05
                    decimals: 2
                    number: bridge.minConfidence
                    onCommitted: function (v) { bridge.minConfidence = v }
                }
            }
        }
    }
}
