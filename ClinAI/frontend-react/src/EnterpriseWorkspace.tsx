import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  App,
  Avatar,
  Badge,
  Button,
  Card,
  Col,
  ConfigProvider,
  Flex,
  Form,
  Input,
  Layout,
  Modal,
  Progress,
  Row,
  Space,
  Steps,
  Typography,
} from 'antd'
import {
  AudioOutlined,
  HeartOutlined,
  PauseOutlined,
  ReloadOutlined,
  SearchOutlined,
  StopOutlined,
  UserOutlined,
} from '@ant-design/icons'
import './enterprise.css'

const { Header, Content } = Layout
const { Title, Text } = Typography
const { TextArea } = Input

type RecordingState = 'idle' | 'recording' | 'paused' | 'processing' | 'ready'

const api = async <T,>(url: string, options?: RequestInit): Promise<T> => {
  const response = await fetch(url, options)
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

const clock = (seconds: number) =>
  `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`

function Workspace() {
  const { message, modal } = App.useApp()
  const [patientId, setPatientId] = useState('')
  const [notes, setNotes] = useState('')
  const [conversation, setConversation] = useState('')
  const [recordingState, setRecordingState] = useState<RecordingState>('idle')
  const [activeStep, setActiveStep] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const [reviewOpen, setReviewOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [lookingUp, setLookingUp] = useState(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    if (recordingState !== 'recording') return
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000)
    return () => window.clearInterval(timer)
  }, [recordingState])

  useEffect(
    () => () => streamRef.current?.getTracks().forEach((track) => track.stop()),
    [],
  )

  const processAudio = async (blob: Blob) => {
    setRecordingState('processing')
    setActiveStep(1)
    try {
      const form = new FormData()
      form.append('file', blob, 'clinical-visit.webm')
      const transcribed = await api<{ transcription?: string }>('/transcribe', {
        method: 'POST',
        body: form,
      })
      const fresh = transcribed.transcription?.trim()
      if (!fresh) throw new Error('No speech was detected. Please record again.')

      setActiveStep(2)
      const combined = conversation ? `${conversation}\n${fresh}` : fresh
      const labeled = await api<{ labeled_conversation?: string }>('/label_conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation: combined }),
      })
      setConversation(labeled.labeled_conversation || combined)
      setActiveStep(3)
      setRecordingState('ready')
      setReviewOpen(true)
      message.success('Transcript ready for clinical review')
    } catch (caught) {
      setRecordingState('idle')
      setActiveStep(0)
      setError(caught instanceof Error ? caught.message : 'Audio processing failed.')
    }
  }

  const startRecording = async () => {
    setError('')
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error('Secure audio recording is not supported in this browser.')
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      streamRef.current = stream
      recorderRef.current = recorder
      chunksRef.current = []
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        void processAudio(blob)
      }
      recorder.start(1000)
      setElapsed(0)
      setActiveStep(0)
      setRecordingState('recording')
    } catch (caught) {
      setError(
        caught instanceof DOMException && caught.name === 'NotAllowedError'
          ? 'Microphone permission was denied. Allow access and try again.'
          : caught instanceof Error
            ? caught.message
            : 'Unable to start microphone.',
      )
    }
  }

  const pauseRecording = () => {
    const recorder = recorderRef.current
    if (!recorder) return
    if (recorder.state === 'recording') {
      recorder.pause()
      setRecordingState('paused')
    } else if (recorder.state === 'paused') {
      recorder.resume()
      setRecordingState('recording')
    }
  }

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
    }
  }

  const lookupPatient = async () => {
    if (!patientId.trim()) {
      setError('Enter a patient ID before searching.')
      return
    }
    setError('')
    setLookingUp(true)
    try {
      await api(`/api/patient/${encodeURIComponent(patientId.trim())}`)
      message.success(`Patient ${patientId.trim()} found`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Patient lookup failed.')
    } finally {
      setLookingUp(false)
    }
  }

  const saveRecord = async () => {
    if (!patientId.trim() || !conversation.trim()) {
      setError('Patient ID and a conversation transcript are required.')
      return
    }
    setSaving(true)
    try {
      await api('/save_record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idx: patientId.trim(), conversation, notes }),
      })
      setReviewOpen(false)
      message.success('Clinical record saved and indexed')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to save the record.')
    } finally {
      setSaving(false)
    }
  }

  const reset = () => {
    const clear = () => {
      setPatientId('')
      setNotes('')
      setConversation('')
      setElapsed(0)
      setRecordingState('idle')
      setActiveStep(0)
      setError('')
    }
    if (!notes && !conversation) return clear()
    modal.confirm({
      title: 'Discard this encounter?',
      content: 'The unsaved transcript and clinician notes will be removed.',
      okText: 'Discard encounter',
      okButtonProps: { danger: true },
      onOk: clear,
    })
  }

  const stateText = {
    idle: 'Ready to record',
    recording: 'Recording conversation',
    paused: 'Recording paused',
    processing: 'Processing clinical audio',
    ready: 'Ready for review',
  }[recordingState]

  return (
    <Layout className="enterprise-shell">
      <Header className="enterprise-header">
        <Flex align="center" justify="space-between">
          <Flex align="center" gap={12}>
            <Avatar shape="square" size={38} className="brand-avatar" icon={<HeartOutlined />} />
            <div className="brand-copy">
              <Text strong>ClinAI</Text>
              <Text type="secondary">Voice notes</Text>
            </div>
          </Flex>
          <Space size={12}>
            <Button icon={<ReloadOutlined />} onClick={reset}>New encounter</Button>
            <Button type="text" href="/">Exit</Button>
          </Space>
        </Flex>
      </Header>

      <Content className="enterprise-content compact-content">
        <Flex align="center" justify="space-between" className="compact-title-row">
          <Title level={2}>New voice note</Title>
          <Badge
            status={recordingState === 'recording' ? 'processing' : recordingState === 'processing' ? 'warning' : 'success'}
            text={stateText}
          />
        </Flex>

        <Card className="patient-bar">
          <Flex align="center" gap={10}>
            <Input
              size="large"
              prefix={<UserOutlined />}
              value={patientId}
              onChange={(event) => setPatientId(event.target.value)}
              placeholder="Patient ID"
              onPressEnter={lookupPatient}
            />
            <Button size="large" icon={<SearchOutlined />} loading={lookingUp} onClick={lookupPatient}>
              Find
            </Button>
          </Flex>
        </Card>

        {error && <Alert closable showIcon type="error" message={error} onClose={() => setError('')} className="error-alert" />}

        <Row gutter={[20, 20]} className="capture-grid">
          <Col xs={24} lg={10}>
            <Card className="workspace-card notes-card" title="Notes">
              <TextArea
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="Type notes here…"
                autoSize={{ minRows: 16, maxRows: 16 }}
              />
            </Card>
          </Col>

          <Col xs={24} lg={14}>
            <Card className={`workspace-card voice-card ${recordingState}`}>
              <div className="voice-console">
                <Title className="timer">{clock(elapsed)}</Title>
                <Text className="voice-state">{stateText}</Text>

                {recordingState === 'idle' || recordingState === 'ready' ? (
                  <button className="primary-mic" onClick={startRecording} aria-label="Start recording">
                    <AudioOutlined />
                    <span>{conversation ? 'Record more' : 'Tap to record'}</span>
                  </button>
                ) : recordingState === 'recording' || recordingState === 'paused' ? (
                  <>
                    <div className={`antd-waveform ${recordingState}`} aria-hidden="true">
                      {Array.from({ length: 42 }, (_, index) => <i key={index} style={{ animationDelay: `${(index % 9) * 70}ms` }} />)}
                    </div>
                    <Space size={12}>
                      <Button ghost size="large" icon={<PauseOutlined />} onClick={pauseRecording}>
                        {recordingState === 'paused' ? 'Resume' : 'Pause'}
                      </Button>
                      <Button danger size="large" icon={<StopOutlined />} onClick={stopRecording}>Stop</Button>
                    </Space>
                  </>
                ) : (
                  <Flex vertical align="center" gap={14}>
                    <Progress type="circle" size={64} percent={activeStep === 1 ? 45 : 78} showInfo={false} status="active" />
                    <Text className="processing-copy">Creating note…</Text>
                  </Flex>
                )}
              </div>
            </Card>
          </Col>
        </Row>

        <Flex align="center" justify="space-between" className="compact-footer">
          <Steps
            size="small"
            current={activeStep}
            items={[
              { title: 'Record' },
              { title: 'Transcribe' },
              { title: 'Extract' },
              { title: 'Review' },
            ]}
          />
          <Button
            type="primary"
            size="large"
            disabled={!conversation || recordingState === 'processing'}
            onClick={() => setReviewOpen(true)}
          >
            Review note
          </Button>
        </Flex>
      </Content>

      <Modal
        title="Review clinical documentation"
        open={reviewOpen}
        width={760}
        okText="Approve and save record"
        cancelText="Continue editing"
        confirmLoading={saving}
        onOk={saveRecord}
        onCancel={() => setReviewOpen(false)}
      >
        <Alert
          showIcon
          type="info"
          message="Human review required"
          description="Verify AI-generated content against the original conversation before saving."
          className="review-alert"
        />
        <Form layout="vertical">
          <Form.Item label="Patient ID" required>
            <Input size="large" value={patientId} onChange={(event) => setPatientId(event.target.value)} />
          </Form.Item>
          <Form.Item label="Speaker-labeled transcript" required>
            <TextArea rows={12} value={conversation} onChange={(event) => setConversation(event.target.value)} />
          </Form.Item>
        </Form>
      </Modal>
    </Layout>
  )
}

export function EnterpriseWorkspace() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1677ff',
          colorInfo: '#1677ff',
          colorSuccess: '#12a474',
          borderRadius: 8,
          fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          colorBgLayout: '#f5f7fa',
        },
        components: {
          Layout: { headerBg: '#ffffff', siderBg: '#ffffff' },
          Card: { headerFontSize: 15 },
          Button: { fontWeight: 600 },
        },
      }}
    >
      <App>
        <Workspace />
      </App>
    </ConfigProvider>
  )
}
