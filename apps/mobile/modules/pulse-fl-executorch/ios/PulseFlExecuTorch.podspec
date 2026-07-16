Pod::Spec.new do |s|
  s.name           = 'PulseFlExecuTorch'
  s.version        = '1.0.0'
  s.summary        = 'ExecuTorch client native module for Better-Pulse.'
  s.description    = 'Integrates ExecuTorch C++ runtime into Expo for Arrhythmia Detection.'
  s.homepage       = 'https://github.com/bernardbdas/better-pulse'
  s.license        = { :type => 'MIT' }
  s.author         = '@bernardbdas'
  s.source         = { :git => 'https://github.com/bernardbdas/better-pulse.git' }
  s.platform       = :ios, '13.4'
  s.swift_version  = '5.4'
  s.source_files   = '**/*.{h,m,swift}'

  s.dependency 'ExpoModulesCore'

  # Note: ExecuTorch C++ SDK libraries can be added here as dependencies, 
  # e.g., s.dependency 'executorch-cpp-framework' or via local vendored libraries.
end
