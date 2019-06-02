import { Injectable } from '@angular/core';
import {BaseService} from "../../../base.service";

@Injectable({
  providedIn: 'root'
})
export class ImgClassifyService extends BaseService{
   protected collection_part: string = 'img_classify';
   protected single_part: string;

}
