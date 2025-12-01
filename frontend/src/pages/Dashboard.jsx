import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import { policyAPI, moderationAPI } from '../services/api';
import { 
  Upload, FileText, History, AlertCircle, CheckCircle, Clock, Shield, 
  Trash2, LayoutDashboard, Settings, FileCheck, X, Search, Filter,
  ChevronDown, Download, Eye, MoreVertical
} from 'lucide-react';
import { dashboardStyles as styles } from '../styles/dashboard.styles';

const Dashboard = () => {
  const [activeView, setActiveView] = useState('overview');
  const [policyFiles, setPolicyFiles] = useState([]);
  const [moderateFile, setModerateFile] = useState(null);
  const [policies, setPolicies] = useState([]);
  const [history, setHistory] = useState([]);
  const [policyLoading, setPolicyLoading] = useState(false);
  const [moderationLoading, setModerationLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [viewingFile, setViewingFile] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [moderationResult, setModerationResult] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [stats, setStats] = useState({
    totalPolicies: 0,
    totalFiles: 0,
    filesApproved: 0,      // verdict = 'clean'
    filesViolation: 0,     // verdict = 'violation_found'
    filesManualReview: 0,  // verdict = 'needs_review'
  });

  useEffect(() => {
    loadPolicies();
    loadHistory();
  }, []);

  useEffect(() => {
    if (history.length > 0) {
      // Count files by their AI verdict (3 states)
      const approved = history.filter(h => h.verdict === 'clean').length;
      const violation = history.filter(h => h.verdict === 'violation_found').length;
      const manualReview = history.filter(h => h.verdict === 'needs_review').length;
      
      setStats({
        totalPolicies: policies.length,
        totalFiles: history.length,
        filesApproved: approved,
        filesViolation: violation,
        filesManualReview: manualReview,
      });
    } else {
      setStats({
        totalPolicies: policies.length,
        totalFiles: 0,
        filesApproved: 0,
        filesViolation: 0,
        filesManualReview: 0,
      });
    }
  }, [history, policies]);

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  const loadPolicies = async () => {
    try {
      const response = await policyAPI.listPolicies();
      setPolicies(response.data.policies);
    } catch (error) {
      console.error('Failed to load policies');
    }
  };

  const loadHistory = async () => {
    try {
      const response = await moderationAPI.getHistory();
      setHistory(response.data.results);
    } catch (error) {
      console.error('Failed to load history');
    }
  };

  const handlePolicyUpload = async (e) => {
    e.preventDefault();
    if (!policyFiles.length) {
      showMessage('error', 'Please select at least one policy file');
      return;
    }

    setPolicyLoading(true);
    try {
      await policyAPI.uploadPolicies(policyFiles);
      showMessage('success', 'Policies uploaded successfully');
      setPolicyFiles([]);
      e.target.reset();
      loadPolicies();
    } catch (error) {
      showMessage('error', error.response?.data?.error || 'Failed to upload policies');
    } finally {
      setPolicyLoading(false);
    }
  };

  const handleModeration = async (e) => {
    e.preventDefault();
    if (!moderateFile) {
      showMessage('error', 'Please select a file to moderate');
      return;
    }

    setModerationLoading(true);
    setModerationResult(null);
    try {
      const response = await moderationAPI.moderateFile(moderateFile);
      setModerationResult(response.data);
      showMessage('success', 'Content moderated successfully');
      setModerateFile(null);
      e.target.reset();
      loadHistory();
      setActiveView('submissions');
    } catch (error) {
      showMessage('error', error.response?.data?.error || 'Failed to moderate file');
    } finally {
      setModerationLoading(false);
    }
  };

  const handleClearPolicies = async () => {
    if (!window.confirm('This will remove all policies from the system. Continue?')) return;

    setPolicyLoading(true);
    try {
      await policyAPI.clearPolicies();
      showMessage('success', 'All policies cleared');
      setPolicies([]);
      loadPolicies();
    } catch (error) {
      showMessage('error', 'Failed to clear policies');
    } finally {
      setPolicyLoading(false);
    }
  };

  const handleFinalVerdict = async (resultId, verdict) => {
    try {
      await moderationAPI.updateFinalVerdict(resultId, verdict);
      showMessage('success', `Verdict updated: ${verdict === 'approved' ? 'Approved' : 'Rejected'}`);
      loadHistory();
      if (selectedResult && selectedResult.id === resultId) {
        setSelectedResult(null);
      }
      if (moderationResult && moderationResult.id === resultId) {
        setModerationResult(null);
      }
    } catch (error) {
      showMessage('error', 'Failed to update verdict');
    }
  };

  const handleViewDetails = async (resultId) => {
    try {
      const response = await moderationAPI.getDetail(resultId);
      setSelectedResult(response.data);
      setViewingFile(true);
    } catch (error) {
      showMessage('error', 'Failed to load details');
    }
  };

  const filteredHistory = history.filter(item => {
    const matchesSearch = item.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === 'all' || 
      (filterStatus === 'pending' && item.final_verdict === 'pending') ||
      (filterStatus === 'approved' && item.final_verdict === 'approved') ||
      (filterStatus === 'rejected' && item.final_verdict === 'rejected');
    return matchesSearch && matchesFilter;
  });

  return (
    <div style={styles.container}>
      <Navbar />
      
      <div style={styles.layout}>
        {/* Sidebar */}
        <aside style={styles.sidebar}>
          <nav style={styles.sidebarNav}>
            <button
              onClick={() => setActiveView('overview')}
              style={{...styles.navItem, ...(activeView === 'overview' ? styles.navItemActive : {})}}
            >
              <LayoutDashboard size={20} />
              <span>Overview</span>
            </button>
            <button
              onClick={() => setActiveView('moderate')}
              style={{...styles.navItem, ...(activeView === 'moderate' ? styles.navItemActive : {})}}
            >
              <FileCheck size={20} />
              <span>Moderate Content</span>
            </button>
            <button
              onClick={() => setActiveView('submissions')}
              style={{...styles.navItem, ...(activeView === 'submissions' ? styles.navItemActive : {})}}
            >
              <History size={20} />
              <span>Submissions</span>
              {stats.pendingReview > 0 && (
                <span style={styles.badge}>{stats.pendingReview}</span>
              )}
            </button>
            <button
              onClick={() => setActiveView('policies')}
              style={{...styles.navItem, ...(activeView === 'policies' ? styles.navItemActive : {})}}
            >
              <Shield size={20} />
              <span>Policies</span>
            </button>
          </nav>
        </aside>

        {/* Main Content */}
        <main style={styles.main}>
          {message.text && (
            <div style={{
              ...styles.alert,
              backgroundColor: message.type === 'success' ? '#ecfdf5' : '#fef2f2',
              borderColor: message.type === 'success' ? '#10b981' : '#ef4444',
              color: message.type === 'success' ? '#065f46' : '#991b1b',
            }}>
              {message.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
              <span>{message.text}</span>
              <button onClick={() => setMessage({ type: '', text: '' })} style={styles.alertClose}>
                <X size={16} />
              </button>
            </div>
          )}

          {/* Overview */}
          {activeView === 'overview' && (
            <div>
              <div style={styles.pageHeader}>
                <div>
                  <h1 style={styles.pageTitle}>Dashboard Overview</h1>
                  <p style={styles.pageSubtitle}>Monitor your content moderation metrics and activity</p>
                </div>
              </div>

              <div style={styles.statsGrid}>
                <div style={styles.statCard}>
                  <div style={styles.statCardHeader}>
                    <span style={styles.statCardTitle}>Total Files</span>
                    <LayoutDashboard size={20} style={{color: '#6b7280'}} />
                  </div>
                  <div style={styles.statCardValue}>{stats.totalFiles}</div>
                  <div style={styles.statCardFooter}>
                    <span style={{color: '#6b7280'}}>All submissions</span>
                  </div>
                </div>

                <div style={styles.statCard}>
                  <div style={styles.statCardHeader}>
                    <span style={styles.statCardTitle}>Approved</span>
                    <CheckCircle size={20} style={{color: '#6b7280'}} />
                  </div>
                  <div style={styles.statCardValue}>{stats.filesApproved}</div>
                  <div style={styles.statCardFooter}>
                    <span style={{color: '#10b981'}}>Clean files</span>
                  </div>
                </div>

                <div style={styles.statCard}>
                  <div style={styles.statCardHeader}>
                    <span style={styles.statCardTitle}>Violations</span>
                    <AlertCircle size={20} style={{color: '#6b7280'}} />
                  </div>
                  <div style={styles.statCardValue}>{stats.filesViolation}</div>
                  <div style={styles.statCardFooter}>
                    <span style={{color: '#ef4444'}}>Policy violations</span>
                  </div>
                </div>

                <div style={styles.statCard}>
                  <div style={styles.statCardHeader}>
                    <span style={styles.statCardTitle}>Manual Review</span>
                    <Clock size={20} style={{color: '#6b7280'}} />
                  </div>
                  <div style={styles.statCardValue}>{stats.filesManualReview}</div>
                  <div style={styles.statCardFooter}>
                    <span style={{color: '#f59e0b'}}>Needs review</span>
                  </div>
                </div>

                <div style={styles.statCard}>
                  <div style={styles.statCardHeader}>
                    <span style={styles.statCardTitle}>Active Policies</span>
                    <Shield size={20} style={{color: '#6b7280'}} />
                  </div>
                  <div style={styles.statCardValue}>{stats.totalPolicies}</div>
                  <div style={styles.statCardFooter}>
                    <span style={{color: '#8b5cf6'}}>In use</span>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div style={styles.section}>
                <h2 style={styles.sectionTitle}>Quick Actions</h2>
                <div style={styles.quickActions}>
                  <button onClick={() => setActiveView('moderate')} style={styles.quickActionCard}>
                    <FileCheck size={32} style={{color: '#8b5cf6'}} />
                    <h3 style={styles.quickActionTitle}>Moderate Content</h3>
                    <p style={styles.quickActionDesc}>Submit new content for policy review</p>
                  </button>
                  <button onClick={() => setActiveView('policies')} style={styles.quickActionCard}>
                    <Shield size={32} style={{color: '#8b5cf6'}} />
                    <h3 style={styles.quickActionTitle}>Manage Policies</h3>
                    <p style={styles.quickActionDesc}>Upload and configure policy documents</p>
                  </button>
                  <button onClick={() => setActiveView('submissions')} style={styles.quickActionCard}>
                    <History size={32} style={{color: '#8b5cf6'}} />
                    <h3 style={styles.quickActionTitle}>View Submissions</h3>
                    <p style={styles.quickActionDesc}>Review moderation history and results</p>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Moderate Content */}
          {activeView === 'moderate' && (
            <div>
              <div style={styles.pageHeader}>
                <div>
                  <h1 style={styles.pageTitle}>Moderate Content</h1>
                  <p style={styles.pageSubtitle}>Submit content for automated policy compliance review</p>
                </div>
              </div>

              <div style={styles.contentCard}>
                <form onSubmit={handleModeration}>
                  <div style={styles.formGroup}>
                    <label style={styles.formLabel}>Upload Document</label>
                    <div style={styles.uploadZone}>
                      <input
                        type="file"
                        accept=".pdf"
                        onChange={(e) => setModerateFile(e.target.files[0])}
                        style={styles.hiddenInput}
                        id="moderate-file"
                      />
                      <label htmlFor="moderate-file" style={styles.uploadZoneLabel}>
                        <Upload size={48} style={{color: '#9ca3af'}} />
                        <p style={styles.uploadZoneText}>
                          {moderateFile ? moderateFile.name : 'Click to upload or drag and drop'}
                        </p>
                        <p style={styles.uploadZoneHint}>PDF files up to 10MB</p>
                      </label>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={moderationLoading || !moderateFile}
                    style={{
                      ...styles.primaryButton,
                      opacity: moderationLoading || !moderateFile ? 0.5 : 1,
                    }}
                  >
                    {moderationLoading ? 'Analyzing...' : 'Submit for Moderation'}
                  </button>
                </form>

                {moderationResult && (
                  <div style={styles.resultSection}>
                    <div style={styles.resultHeader}>
                      <h3 style={styles.resultTitle}>Moderation Complete</h3>
                      <div style={{
                        ...styles.statusBadge,
                        backgroundColor: 
                          moderationResult.verdict === 'clean' ? '#ecfdf5' : 
                          moderationResult.verdict === 'needs_review' ? '#fef3c7' : '#fef2f2',
                        color: 
                          moderationResult.verdict === 'clean' ? '#065f46' : 
                          moderationResult.verdict === 'needs_review' ? '#92400e' : '#991b1b',
                      }}>
                        {moderationResult.verdict === 'clean' ? 'Clean' : 
                         moderationResult.verdict === 'needs_review' ? 'Needs Review' : 'Violations Found'}
                      </div>
                    </div>

                    <div style={styles.resultGrid}>
                      <div style={styles.resultMetric}>
                        <span style={styles.resultMetricLabel}>Total Chunks</span>
                        <span style={styles.resultMetricValue}>{moderationResult.total_chunks}</span>
                      </div>
                      <div style={styles.resultMetric}>
                        <span style={styles.resultMetricLabel}>Allowed</span>
                        <span style={{...styles.resultMetricValue, color: '#10b981'}}>{moderationResult.allowed_chunks}</span>
                      </div>
                      <div style={styles.resultMetric}>
                        <span style={styles.resultMetricLabel}>Review</span>
                        <span style={{...styles.resultMetricValue, color: '#f59e0b'}}>{moderationResult.review_chunks}</span>
                      </div>
                      <div style={styles.resultMetric}>
                        <span style={styles.resultMetricLabel}>Violations</span>
                        <span style={{...styles.resultMetricValue, color: '#ef4444'}}>{moderationResult.violation_chunks}</span>
                      </div>
                    </div>

                    {moderationResult.violations && moderationResult.violations.length > 0 && (
                      <div style={styles.violationsList}>
                        <h4 style={styles.violationsListTitle}>Flagged Content</h4>
                        {moderationResult.violations.slice(0, 3).map((violation, idx) => (
                          <div key={idx} style={styles.violationItem}>
                            <div style={styles.violationItemHeader}>
                              <span style={{
                                ...styles.violationBadge,
                                backgroundColor: violation.verdict === 'violation' ? '#fef2f2' : '#fef3c7',
                                color: violation.verdict === 'violation' ? '#991b1b' : '#92400e',
                              }}>
                                {violation.verdict.toUpperCase()}
                              </span>
                            </div>
                            <p style={styles.violationText}>{violation.chunk_text.substring(0, 200)}...</p>
                            <p style={styles.violationExplanation}>{violation.explanation}</p>
                          </div>
                        ))}
                        {moderationResult.violations.length > 3 && (
                          <p style={styles.moreViolations}>
                            +{moderationResult.violations.length - 3} more issues
                          </p>
                        )}
                      </div>
                    )}

                    {moderationResult.final_verdict === 'pending' && (
                      <div style={styles.actionSection}>
                        <h4 style={styles.actionSectionTitle}>Make Final Decision</h4>
                        {moderationResult.file_url && (
                          <a 
                            href={moderationResult.file_url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            style={styles.viewFileLink}
                          >
                            <Eye size={18} />
                            View Original File
                          </a>
                        )}
                        <div style={styles.actionButtons}>
                          <button
                            onClick={() => handleFinalVerdict(moderationResult.id, 'approved')}
                            style={styles.approveButton}
                          >
                            <CheckCircle size={20} />
                            Approve
                          </button>
                          <button
                            onClick={() => handleFinalVerdict(moderationResult.id, 'rejected')}
                            style={styles.rejectButton}
                          >
                            <AlertCircle size={20} />
                            Reject
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Submissions */}
          {activeView === 'submissions' && (
            <div>
              <div style={styles.pageHeader}>
                <div>
                  <h1 style={styles.pageTitle}>Submissions</h1>
                  <p style={styles.pageSubtitle}>Review and manage all moderated content</p>
                </div>
              </div>

              <div style={styles.contentCard}>
                <div style={styles.tableHeader}>
                  <div style={styles.searchBar}>
                    <Search size={20} style={{color: '#9ca3af'}} />
                    <input
                      type="text"
                      placeholder="Search by filename..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      style={styles.searchInput}
                    />
                  </div>
                  <div style={styles.filterGroup}>
                    <Filter size={16} style={{color: '#6b7280'}} />
                    <select
                      value={filterStatus}
                      onChange={(e) => setFilterStatus(e.target.value)}
                      style={styles.filterSelect}
                    >
                      <option value="all">All Status</option>
                      <option value="pending">Pending Review</option>
                      <option value="approved">Approved</option>
                      <option value="rejected">Rejected</option>
                    </select>
                  </div>
                </div>

                {filteredHistory.length === 0 ? (
                  <div style={styles.emptyState}>
                    <History size={64} style={{color: '#d1d5db'}} />
                    <p style={styles.emptyStateText}>No submissions found</p>
                    <p style={styles.emptyStateHint}>
                      {searchQuery || filterStatus !== 'all' 
                        ? 'Try adjusting your search or filter'
                        : 'Submit content to see moderation results here'}
                    </p>
                  </div>
                ) : (
                  <table style={styles.table}>
                    <thead>
                      <tr style={styles.tableHeaderRow}>
                        <th style={styles.tableHeaderCell}>Filename</th>
                        <th style={styles.tableHeaderCell}>Status</th>
                        <th style={styles.tableHeaderCell}>Decision</th>
                        <th style={styles.tableHeaderCell}>Chunks</th>
                        <th style={styles.tableHeaderCell}>Submitted</th>
                        <th style={styles.tableHeaderCell}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredHistory.map((item) => (
                        <tr key={item.id} style={styles.tableRow}>
                          <td style={styles.tableCell}>
                            <div style={styles.fileCell}>
                              <FileText size={18} style={{color: '#8b5cf6'}} />
                              <span style={styles.filename}>{item.filename}</span>
                            </div>
                          </td>
                          <td style={styles.tableCell}>
                            <span style={{
                              ...styles.tableBadge,
                              backgroundColor: 
                                item.verdict === 'clean' ? '#ecfdf5' : 
                                item.verdict === 'needs_review' ? '#fef3c7' : '#fef2f2',
                              color: 
                                item.verdict === 'clean' ? '#065f46' : 
                                item.verdict === 'needs_review' ? '#92400e' : '#991b1b',
                            }}>
                              {item.verdict === 'clean' ? 'Clean' : 
                               item.verdict === 'needs_review' ? 'Review' : 'Violations'}
                            </span>
                          </td>
                          <td style={styles.tableCell}>
                            <span style={{
                              ...styles.tableBadge,
                              backgroundColor: 
                                item.final_verdict === 'approved' ? '#ecfdf5' :
                                item.final_verdict === 'rejected' ? '#fef2f2' : '#fef3c7',
                              color: 
                                item.final_verdict === 'approved' ? '#065f46' :
                                item.final_verdict === 'rejected' ? '#991b1b' : '#92400e',
                            }}>
                              {item.final_verdict === 'pending' ? 'Pending' :
                               item.final_verdict === 'approved' ? 'Approved' : 'Rejected'}
                            </span>
                          </td>
                          <td style={styles.tableCell}>
                            <div style={styles.chunkMetrics}>
                              <span style={{color: '#10b981'}}>{item.allowed_chunks}</span>
                              <span style={{color: '#9ca3af'}}>/</span>
                              <span style={{color: '#ef4444'}}>{item.violation_chunks}</span>
                              <span style={{color: '#9ca3af'}}>/</span>
                              <span style={{color: '#f59e0b'}}>{item.review_chunks}</span>
                            </div>
                          </td>
                          <td style={styles.tableCell}>
                            <span style={styles.dateText}>
                              {new Date(item.created_at).toLocaleDateString()}
                            </span>
                          </td>
                          <td style={styles.tableCell}>
                            <button
                              onClick={() => handleViewDetails(item.id)}
                              style={styles.actionButton}
                            >
                              <Eye size={16} />
                              View
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {/* Policies */}
          {activeView === 'policies' && (
            <div>
              <div style={styles.pageHeader}>
                <div>
                  <h1 style={styles.pageTitle}>Policy Management</h1>
                  <p style={styles.pageSubtitle}>Upload and manage your content moderation policies</p>
                </div>
                {policies.length > 0 && (
                  <button onClick={handleClearPolicies} style={styles.dangerButton} disabled={policyLoading}>
                    <Trash2 size={18} />
                    Clear All Policies
                  </button>
                )}
              </div>

              <div style={styles.contentCard}>
                <form onSubmit={handlePolicyUpload}>
                  <div style={styles.formGroup}>
                    <label style={styles.formLabel}>Upload Policy Documents</label>
                    <div style={styles.uploadZone}>
                      <input
                        type="file"
                        multiple
                        accept=".pdf"
                        onChange={(e) => setPolicyFiles(Array.from(e.target.files))}
                        style={styles.hiddenInput}
                        id="policy-files"
                      />
                      <label htmlFor="policy-files" style={styles.uploadZoneLabel}>
                        <Upload size={48} style={{color: '#9ca3af'}} />
                        <p style={styles.uploadZoneText}>
                          {policyFiles.length > 0 
                            ? `${policyFiles.length} file(s) selected` 
                            : 'Click to upload or drag and drop'}
                        </p>
                        <p style={styles.uploadZoneHint}>PDF files, multiple selection supported</p>
                      </label>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={policyLoading || policyFiles.length === 0}
                    style={{
                      ...styles.primaryButton,
                      opacity: policyLoading || policyFiles.length === 0 ? 0.5 : 1,
                    }}
                  >
                    {policyLoading ? 'Uploading...' : 'Upload Policies'}
                  </button>
                </form>

                {policies.length > 0 && (
                  <div style={styles.policyList}>
                    <h3 style={styles.policyListTitle}>Active Policies ({policies.length})</h3>
                    {policies.map((policy) => (
                      <div key={policy.id} style={styles.policyItem}>
                        <Shield size={20} style={{color: '#8b5cf6'}} />
                        <div style={styles.policyDetails}>
                          <span style={styles.policyName}>{policy.filename}</span>
                          <span style={styles.policyMeta}>
                            {(policy.file_size / 1024).toFixed(2)} KB • 
                            Uploaded {new Date(policy.uploaded_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Detail Modal */}
      {viewingFile && selectedResult && (
        <div style={styles.modal} onClick={() => setViewingFile(false)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <div>
                <h2 style={styles.modalTitle}>{selectedResult.filename}</h2>
                <p style={styles.modalSubtitle}>Detailed moderation report</p>
              </div>
              <button onClick={() => setViewingFile(false)} style={styles.closeButton}>
                <X size={24} />
              </button>
            </div>
            
            <div style={styles.modalBody}>
              <div style={styles.modalStats}>
                <div style={styles.modalStatItem}>
                  <span style={styles.modalStatLabel}>Status</span>
                  <span style={{
                    ...styles.tableBadge,
                    backgroundColor: 
                      selectedResult.verdict === 'clean' ? '#ecfdf5' : 
                      selectedResult.verdict === 'needs_review' ? '#fef3c7' : '#fef2f2',
                    color: 
                      selectedResult.verdict === 'clean' ? '#065f46' : 
                      selectedResult.verdict === 'needs_review' ? '#92400e' : '#991b1b',
                  }}>
                    {selectedResult.verdict === 'clean' ? 'Clean' : 
                     selectedResult.verdict === 'needs_review' ? 'Needs Review' : 'Violations Found'}
                  </span>
                </div>
                <div style={styles.modalStatItem}>
                  <span style={styles.modalStatLabel}>Total Chunks</span>
                  <span style={styles.modalStatValue}>{selectedResult.total_chunks}</span>
                </div>
                <div style={styles.modalStatItem}>
                  <span style={styles.modalStatLabel}>Allowed</span>
                  <span style={{...styles.modalStatValue, color: '#10b981'}}>{selectedResult.allowed_chunks}</span>
                </div>
                <div style={styles.modalStatItem}>
                  <span style={styles.modalStatLabel}>Review</span>
                  <span style={{...styles.modalStatValue, color: '#f59e0b'}}>{selectedResult.review_chunks}</span>
                </div>
                <div style={styles.modalStatItem}>
                  <span style={styles.modalStatLabel}>Violations</span>
                  <span style={{...styles.modalStatValue, color: '#ef4444'}}>{selectedResult.violation_chunks}</span>
                </div>
              </div>

              {selectedResult.violations && selectedResult.violations.length > 0 && (
                <div style={styles.modalSection}>
                  <h3 style={styles.modalSectionTitle}>Flagged Content</h3>
                  {selectedResult.violations.map((violation, idx) => (
                    <div key={idx} style={styles.modalViolationItem}>
                      <div style={styles.modalViolationHeader}>
                        <span style={{
                          ...styles.violationBadge,
                          backgroundColor: violation.verdict === 'violation' ? '#fef2f2' : '#fef3c7',
                          color: violation.verdict === 'violation' ? '#991b1b' : '#92400e',
                        }}>
                          {violation.verdict.toUpperCase()}
                        </span>
                      </div>
                      <p style={styles.modalViolationText}>{violation.chunk_text}</p>
                      <p style={styles.modalViolationExplanation}>{violation.explanation}</p>
                    </div>
                  ))}
                </div>
              )}

              {selectedResult.final_verdict === 'pending' && (
                <div style={styles.modalActionSection}>
                  <h3 style={styles.modalSectionTitle}>Make Final Decision</h3>
                  {selectedResult.file_url && (
                    <a 
                      href={selectedResult.file_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={styles.viewFileLink}
                    >
                      <Eye size={18} />
                      View Original File
                    </a>
                  )}
                  <div style={styles.actionButtons}>
                    <button
                      onClick={() => {
                        handleFinalVerdict(selectedResult.id, 'approved');
                        setViewingFile(false);
                      }}
                      style={styles.approveButton}
                    >
                      <CheckCircle size={20} />
                      Approve
                    </button>
                    <button
                      onClick={() => {
                        handleFinalVerdict(selectedResult.id, 'rejected');
                        setViewingFile(false);
                      }}
                      style={styles.rejectButton}
                    >
                      <AlertCircle size={20} />
                      Reject
                    </button>
                  </div>
                </div>
              )}

              {selectedResult.final_verdict !== 'pending' && (
                <div style={styles.modalFinalVerdict}>
                  <span style={{
                    ...styles.tableBadge,
                    backgroundColor: selectedResult.final_verdict === 'approved' ? '#ecfdf5' : '#fef2f2',
                    color: selectedResult.final_verdict === 'approved' ? '#065f46' : '#991b1b',
                  }}>
                    Final Decision: {selectedResult.final_verdict === 'approved' ? 'Approved' : 'Rejected'}
                  </span>
                  <span style={styles.dateText}>
                    {new Date(selectedResult.reviewed_at).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;