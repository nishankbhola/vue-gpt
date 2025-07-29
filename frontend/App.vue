<template>
  <div id="app">
    <div class="app-container">
      <div class="sidebar">
        <div class="sidebar-header">
          <div class="logo-section">
            <div class="logo-icon">BI</div>
            <div>
              <div class="app-title">BIBLIO</div>
              <div class="app-subtitle">Insurance Assistant</div>
            </div>
          </div>
        </div>
        
        <div class="sidebar-content">
          <div class="section-title">
            <i class="fas fa-building"></i> Companies
          </div>
          
          <div v-if="companies.length === 0" class="alert alert-warning">
            <i class="fas fa-exclamation-triangle"></i>
            <span>BIBLIO is under maintenance</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      companies: []
    };
  },
  mounted() {
    // Logic to load companies or handle maintenance state
  }
};
</script>

<style scoped>
/* Add styles here */
</style>
