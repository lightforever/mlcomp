import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ImgClassifyComponent } from './img-classify.component';

describe('ImgClassifyComponent', () => {
  let component: ImgClassifyComponent;
  let fixture: ComponentFixture<ImgClassifyComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ImgClassifyComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ImgClassifyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
